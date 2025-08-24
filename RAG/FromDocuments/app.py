import streamlit as st
import sys
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# SQLite compatibility shim for Streamlit Cloud (use pysqlite3 if available)
try:
    import pysqlite3 as _sqlite3  # type: ignore
    sys.modules["sqlite3"] = _sqlite3
    sys.modules["_sqlite3"] = _sqlite3
except Exception:
    pass
try:
    from langchain_chroma import Chroma as ChromaStore
    CHROMA_DEPRECATED = False
except Exception:
    from langchain_community.vectorstores import Chroma as ChromaStore
    CHROMA_DEPRECATED = True
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None

# Pydantic models for structured output
class Expansions(BaseModel):
    items: List[str] = Field(..., min_items=1, description="List of paraphrased questions")
    
    @field_validator('items')
    @classmethod
    def validate_items(cls, v):
        if not all(isinstance(item, str) and len(item.strip()) > 0 for item in v):
            raise ValueError("All items must be non-empty strings")
        return [item.strip() for item in v]

class RetrievedMetadata(BaseModel):
    source: str
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    position: Optional[int] = Field(None, alias="start_index")
    chunk_number: Optional[int] = None
    chunk_summary: Optional[str] = None

class RetrievedChunk(BaseModel):
    question: str
    score: float
    content: str
    metadata: RetrievedMetadata

class RetrievalResults(BaseModel):
    results: List[RetrievedChunk]

class RerankedDocument(BaseModel):
    content: str
    metadata: RetrievedMetadata
    rerank_score: float
    
class RerankedResults(BaseModel):
    results: List[RerankedDocument]
    original_question: str
    model_name: str = "BAAI/bge-reranker-base"

class Citation(BaseModel):
    source: str
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    position: Optional[int] = Field(None, alias="start_index")
    chunk_number: Optional[int] = None

class GeminiResponse(BaseModel):
    answer: str = Field(..., description="The answer from Gemini")
    citations: List[Citation] = Field(default_factory=list, description="Citations from the context")

# Setup page config
st.set_page_config(
    page_title="Alice in Wonderland Q&A",
    page_icon="🎩",
    layout="wide"
)

# Constants
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION = "alice"

def get_api_key_from_secrets() -> str:
    return st.secrets.get("GEMINI_API_KEY", "")

def get_api_key_from_env() -> str:
    return os.getenv("GEMINI_API_KEY", "")

def check_api_key():
    """Check if API key is properly set (prefer Streamlit Secrets by default)."""
    # Default: use Streamlit Secrets
    api_key = get_api_key_from_secrets()

    # Local development option: comment the above and uncomment the next line
    # api_key = get_api_key_from_env()

    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found. Add it in Settings → Secrets (Cloud) or .env (local).")
        return False
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    return True

@st.cache_resource
def load_vectordb():
    """Load the persisted Chroma DB built in the notebook using BGE embeddings."""
    try:
        sqlite_path = os.path.join(PERSIST_DIR, "chroma.sqlite3")
        if not os.path.isdir(PERSIST_DIR) or not os.path.exists(sqlite_path):
            st.error("Chroma store not found. Run the notebook to build 'chroma_store/'.")
            return None
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vectordb = ChromaStore(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION,
        )
        if CHROMA_DEPRECATED:
            st.warning("Using deprecated Chroma import from langchain_community. Consider installing 'langchain-chroma' and switching to the new import.")
        return vectordb
    except Exception as e:
        st.error(f"Error loading Chroma DB: {str(e)}")
        return None

def expand_queries(llm: ChatGoogleGenerativeAI, question: str, n: int = 4) -> List[str]:
    """Generate query expansions using Gemini with structured output."""
    prompt = f"""
    You are helping to search Alice in Wonderland by generating variations of a question.
    Generate {n} different ways to ask this question that will help find relevant passages.
    Include variations with character names (e.g., "Mad Hatter" and "Hatter") and different phrasings.
    
    Return ONLY a valid JSON object with this exact format:
    {{"items": ["paraphrase1", "paraphrase2", ...]}}

    Question: "{question}"

    Tips:
    - Include character name variations
    - Try different event descriptions
    - Consider scene-specific words
    - Mix specific and general phrasings

    Remember: Return ONLY the JSON object, no other text.
    """
    response = llm.invoke(prompt)
    try:
        # Try to parse the response as JSON
        json_str = response.content.strip()
        # Find JSON object if it's embedded in other text
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = json_str[start:end]
        result = json.loads(json_str)
        
        # Validate with Pydantic
        expansions = Expansions(items=result.get('items', []))
        # Ensure we have exactly n items
        items = expansions.items[:n] if len(expansions.items) > n else expansions.items + [question] * (n - len(expansions.items))
        return items
    except Exception:
        # Fallback: return original question repeated
        return [question] * n

def retrieve_candidates(vectordb, queries, per_query_k: int = 5) -> RetrievalResults:
    """Retrieve candidates with structured output. Always returns exactly per_query_k chunks per query."""
    results = []
    seen = set()
    
    # Ensure we have exactly 4 queries
    queries = queries[:4] if len(queries) > 4 else queries + [queries[0]] * (4 - len(queries))
    
    for q in queries:
        query_results = []
        hits = vectordb.similarity_search_with_score(q, k=per_query_k * 2)  # Get extra to handle duplicates
        
        for doc, score in hits:
            key = (doc.metadata.get('start_index'), doc.metadata.get('chunk_number'))
            if key not in seen and len(query_results) < per_query_k:
                seen.add(key)
                chunk = RetrievedChunk(
                    question=q,
                    score=float(score),
                    content=doc.page_content,
                    metadata=RetrievedMetadata(**doc.metadata)
                )
                query_results.append(chunk)
        
        # If we still don't have enough unique results, relax the uniqueness constraint
        if len(query_results) < per_query_k:
            for doc, score in hits:
                if len(query_results) < per_query_k:
                    chunk = RetrievedChunk(
                        question=q,
                        score=float(score),
                        content=doc.page_content,
                        metadata=RetrievedMetadata(**doc.metadata)
                    )
                    query_results.append(chunk)
        
        results.extend(query_results[:per_query_k])  # Ensure exactly per_query_k results per query
    
    return RetrievalResults(results=results)

def rerank_candidates(question: str, candidates: RetrievalResults, top_n: int = 5) -> RerankedResults:
    """Rerank candidates with structured output."""
    if FlagReranker is None:
        # Fallback: use ANN scores (ascending cosine distance)
        sorted_candidates = sorted(candidates.results, key=lambda x: x.score)
        reranked_docs = [
            RerankedDocument(
                content=doc.content,
                metadata=doc.metadata,
                rerank_score=float(doc.score)
            ) for doc in sorted_candidates[:top_n]
        ]
        return RerankedResults(results=reranked_docs, original_question=question)

    reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
    pairs = [[question, doc.content] for doc in candidates.results]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(candidates.results, scores), key=lambda x: x[1], reverse=True)
    
    seen = set()
    top_docs = []
    for doc, rerank_score in reranked:
        sid = doc.metadata.position
        if sid not in seen:
            seen.add(sid)
            reranked_doc = RerankedDocument(
                content=doc.content,
                metadata=doc.metadata,
                rerank_score=float(rerank_score)
            )
            top_docs.append(reranked_doc)
        if len(top_docs) == top_n:
            break
    
    return RerankedResults(results=top_docs, original_question=question)

def build_context(docs: RerankedResults) -> tuple[str, List[Citation]]:
    """Build context with structured output."""
    parts = []
    citations = []
    
    for doc in docs.results:
        m = doc.metadata
        # Add minimal header for context building
        header = f"[Chapter {m.chapter_number} {m.chapter_title} | Position={m.position}]"
        parts.append(header + "\n" + doc.content)
        
        citations.append(Citation(
            source=m.source,
            chapter_number=m.chapter_number,
            chapter_title=m.chapter_title,
            start_index=m.position,
            chunk_number=m.chunk_number
        ))
    
    return "\n\n".join(parts), citations

def get_answer(
    vectordb,
    query: str,
    use_reranker: bool = True,
    per_query_k: int = 5,
    final_top_n: int = 8,  # Fixed at 8 chunks
) -> GeminiResponse:
    """Get answer with structured output."""
    try:
        # Initialize Gemini with more robust settings
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            top_p=0.95,  # Increased for more diverse responses
            top_k=40,    # Increased for more options
            max_output_tokens=1024  # Increased for longer responses
        )

        # 1. Query Expansion
        st.write("🔄 Generating query variations...")
        expansions = expand_queries(llm_gemini, query, n=4)
        queries = [query] + expansions
        st.write(f"Generated {len(expansions)} variations")

        # 2. Initial Retrieval
        st.write("🔍 Searching knowledge base...")
        candidates = retrieve_candidates(vectordb, queries, per_query_k=per_query_k)
        if not candidates.results:
            return GeminiResponse(
                answer="I couldn't find any relevant information in the book about that. Could you try rephrasing your question?",
                citations=[]
            )
        st.write(f"Found {len(candidates.results)} initial matches")

        # 3. Reranking
        st.write("📊 Reranking results...")
        if use_reranker and candidates.results:
            top_docs = rerank_candidates(query, candidates, top_n=min(final_top_n, len(candidates.results)))
        else:
            # Fallback to ANN scores
            sorted_candidates = sorted(candidates.results, key=lambda x: x.score)
            reranked_docs = [
                RerankedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    rerank_score=float(doc.score)
                ) for doc in sorted_candidates[:final_top_n]
            ]
            top_docs = RerankedResults(results=reranked_docs, original_question=query)

        if not top_docs.results:
            return GeminiResponse(
                answer="I found some matches but couldn't rank them properly. Could you try being more specific?",
                citations=[]
            )
        st.write(f"Selected top {len(top_docs.results)} relevant passages")

        # 4. Context Building
        context, citations = build_context(top_docs)
        
        # Enhanced prompt for better answers
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant answering questions about Alice in Wonderland.
            Use ONLY the provided context to answer the question.
            
            Guidelines:
            1. If the context contains the information, provide a clear and complete answer
            2. If the context doesn't contain enough information, say so
            3. Focus on accuracy and completeness
            4. Use 2-3 sentences minimum
            5. Stay under 100 words unless more detail is needed
            6. Only use quotes if specifically asked
            7. Always provide a full, finished answer
            
            Question: {question}
            
            Context:
            {context}
            
            Remember: Base your answer ONLY on the provided context, not general knowledge.
            """
        )

        # 5. Final Answer Generation
        st.write("💭 Generating answer...")
        chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,  # Lower temperature for more focused answers
            top_p=0.85,
            top_k=30,
            max_output_tokens=1024  # Increased for longer responses
        )
        response = chat.invoke(prompt.format(question=query, context=context))
        
        # Validate response
        if not response.content or len(response.content.strip()) < 10:
            return GeminiResponse(
                answer="I found relevant information but had trouble generating a good answer. Please try asking again, perhaps with different wording.",
                citations=citations
            )
        
        return GeminiResponse(
            answer=response.content,
            citations=citations
        )
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return GeminiResponse(
            answer="I encountered an error while processing your question. Please try asking again with different wording.",
            citations=[]
        )

def main():
    st.title("🎩 Alice in Wonderland Q&A")
    st.markdown(
        "Discover answers grounded in the original text of Alice in Wonderland.\n\n"
        "Ask a question in the box below. The app searches a persistent knowledge base built from the book,"
        " expands your query, reranks the most relevant passages, and answers using only the retrieved context."
    )

    # Check API key first
    if not check_api_key():
        return

    # Load persisted vector DB
    with st.spinner("Loading knowledge base (Chroma)..."):
        vectordb = load_vectordb()
        if vectordb is None:
            return

    # Sidebar: instructions and examples
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "- **Type a question** about the story.\n"
            "- **Or click a sample question** to try.\n"
            "- Optionally **show sources** to see which chunks were used."
        )

        st.subheader("Settings")
        show_sources = st.checkbox("Show retrieved sources", value=True)

        st.subheader("Try a sample")
        examples = [
            "Why did Alice follow the White Rabbit?",
            "What happens at the tea party with the Mad Hatter?",
            "How does Alice meet the Mad Hatter?",
            "Describe the Cheshire Cat's appearance under 50 words?",
        ]
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        for ex in examples:
            if st.button(ex):
                st.session_state.user_query = ex
                st.experimental_rerun()  # For compatibility with older Streamlit versions

    # Main interaction
    query = st.text_input(
        "Ask your question",
        key="user_query",
        placeholder="e.g., Why did Alice follow the White Rabbit?",
    )

    if query:
        with st.spinner("Thinking..."):
            response = get_answer(
                vectordb,
                query,
                use_reranker=True,  # Always use reranker
                final_top_n=8,  # Fixed at 8 chunks
            )
        st.markdown("### Question")
        st.write(query)
        st.markdown("### Answer")
        st.write(response.answer)

        if show_sources and response.citations:
            with st.expander("📚 Source Passages"):
                for citation in response.citations:
                    st.markdown(f"**Chapter {citation.chapter_number} - {citation.chapter_title}**")
                    st.markdown(f"*Position {citation.position}, Chunk {citation.chunk_number}*")

if __name__ == "__main__":
    main()