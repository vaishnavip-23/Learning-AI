import streamlit as st
import sys
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
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
    
    @Field.validator('items')
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
    context_headers: List[str] = Field(default_factory=list, description="Headers from the context")

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
    Generate {n} different phrasings of the following question.
    Return ONLY a valid JSON object with this exact format:
    {{"items": ["paraphrase1", "paraphrase2", ...]}}

    Question: "{question}"

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
    """Retrieve candidates with structured output."""
    results = []
    seen = set()
    for q in queries:
        hits = vectordb.similarity_search_with_score(q, k=per_query_k)
        for doc, score in hits:
            key = (doc.metadata.get('start_index'), doc.metadata.get('chunk_number'))
            if key not in seen:
                seen.add(key)
                chunk = RetrievedChunk(
                    question=q,
                    score=float(score),
                    content=doc.page_content,
                    metadata=RetrievedMetadata(**doc.metadata)
                )
                results.append(chunk)
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

def build_context(docs: RerankedResults) -> tuple[str, List[Citation], List[str]]:
    """Build context with structured output."""
    parts = []
    headers = []
    citations = []
    
    for doc in docs.results:
        m = doc.metadata
        header = f"[Source: {m.source} | Chapter {m.chapter_number} {m.chapter_title} | Position={m.position} | Chunk {m.chunk_number}]"
        headers.append(header)
        parts.append(header + "\n" + doc.content)
        
        citations.append(Citation(
            source=m.source,
            chapter_number=m.chapter_number,
            chapter_title=m.chapter_title,
            start_index=m.position,
            chunk_number=m.chunk_number
        ))
    
    return "\n\n".join(parts), citations, headers

def get_answer(
    vectordb,
    query: str,
    use_reranker: bool = True,
    per_query_k: int = 5,
    final_top_n: int = 8,  # Fixed at 8 chunks
) -> GeminiResponse:
    """Get answer with structured output."""
    try:
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
        )
        expansions = expand_queries(llm_gemini, query, n=4)
        queries = [query] + expansions

        candidates = retrieve_candidates(vectordb, queries, per_query_k=per_query_k)
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
                answer="No relevant information found.",
                citations=[],
                context_headers=[]
            )

        context, citations, headers = build_context(top_docs)
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant. 
            Answer the user question using ONLY the provided context.
            Read the chunk summary carefully and if it matches with the question then check the chunk content and answer the question.
            Expand the answer into at least 2–3 sentences and under 60 words and don't use quotes from the content unless the question is asking for the quotes. Always finish the answer and don't leave it in the middle.

            Question: {question}
            Context:
            {context}
            """
        )

        chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            top_p=0.85,
            top_k=30,
            max_output_tokens=512
        )
        response = chat.invoke(prompt.format(question=query, context=context))
        
        return GeminiResponse(
            answer=response.content,
            citations=citations,
            context_headers=headers
        )
    except Exception as e:
        return GeminiResponse(
            answer=f"Error generating answer: {str(e)}",
            citations=[],
            context_headers=[]
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
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

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
            with st.expander("Citations"):
                for citation, header in zip(response.citations, response.context_headers):
                    st.markdown(f"- **Chapter**: {citation.chapter_number} - {citation.chapter_title}")
                    st.markdown(f"  **Source**: {citation.source} | **Position**: {citation.position} | **Chunk**: {citation.chunk_number}")
                    st.markdown(f"  **Header**: {header}")

if __name__ == "__main__":
    main()