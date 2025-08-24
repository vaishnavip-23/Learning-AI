import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
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

# Setup page config
st.set_page_config(
    page_title="Alice in Wonderland Q&A",
    page_icon="🎩",
    layout="wide"
)

# Try to load environment variables from multiple locations
possible_env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),  # Local directory
    os.path.expanduser('~/.env'),  # Home directory
    os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # Parent directory
    '.env'  # Current working directory
]

env_loaded = False
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        st.success(f"Found .env file at: {env_path}")
        break

if not env_loaded:
    st.error("Could not find .env file in any of the expected locations. Please ensure it exists and contains GEMINI_API_KEY.")

# Constants
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION = "alice"

def check_api_key():
    """Check if API key is properly set"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment!")
        st.info("Please ensure your .env file contains: GEMINI_API_KEY=your_key_here")
        return False
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

def expand_queries(llm: ChatGoogleGenerativeAI, question: str, n: int = 4):
    prompt = f"""
    Generate {n} different phrasings of the following user question:
    "{question}"
    Provide only the variations, one per line.
    """
    out = llm.invoke(prompt)
    expansions = list(set(out.content.strip().split("\n")))
    return [q.strip("- ").strip() for q in expansions if q.strip()]

def retrieve_candidates(vectordb, queries, per_query_k: int = 5):
    results = []
    seen = set()
    for q in queries:
        hits = vectordb.similarity_search_with_score(q, k=per_query_k)
        for doc, score in hits:
            key = (doc.metadata.get('start_index'), doc.metadata.get('chunk_number'))
            if key not in seen:
                seen.add(key)
                results.append((doc, score, q))
    return results

def rerank_candidates(question: str, candidates, top_n: int = 5):
    if FlagReranker is None:
        # Fallback: use ANN scores (ascending cosine distance)
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        return [(doc, score) for (doc, score, _q) in sorted_candidates[:top_n]]
    reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
    pairs = [[question, doc.page_content] for doc, _, _ in candidates]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    seen = set()
    top_docs = []
    for (doc, _score, _q), rerank_score in reranked:
        start_index = doc.metadata.get('start_index')
        if start_index not in seen:
            seen.add(start_index)
            top_docs.append((doc, rerank_score))
        if len(top_docs) == top_n:
            break
    return top_docs

def build_context(doc_with_scores):
    parts = []
    for d, _ in doc_with_scores:
        m = getattr(d, 'metadata', {}) or {}
        start_index = m.get('start_index', '?')
        chunk_number = m.get('chunk_number', '?')
        source = m.get('source', 'unknown')
        header = f"[Source: {source} | Position={start_index} | Chunk {chunk_number}]"
        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)

def get_answer(
    vectordb,
    query: str,
    use_reranker: bool = True,
    per_query_k: int = 5,
    final_top_n: int = 5,
):
    try:
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
        )
        expansions = expand_queries(llm_gemini, query, n=4)
        queries = [query] + expansions

        candidates = retrieve_candidates(vectordb, queries, per_query_k=per_query_k)
        if use_reranker and candidates:
            top_docs = rerank_candidates(query, candidates, top_n=min(final_top_n, len(candidates)))
        else:
            sorted_candidates = sorted(candidates, key=lambda x: x[1])
            top_docs = [(doc, score) for (doc, score, _q) in sorted_candidates[:final_top_n]]

        if not top_docs:
            return "No relevant information found.", []

        context = build_context(top_docs)
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
        return response.content, top_docs
    except Exception as e:
        return f"Error generating answer: {str(e)}", []

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
            "- Toggle **Rerank with cross-encoder** for higher relevance.\n"
            "- Optionally **show sources** to see which chunks were used."
        )

        st.subheader("Settings")
        use_reranker = st.checkbox("Rerank with BGE cross-encoder", value=True)
        final_top_n = st.slider("Max supporting chunks", 3, 8, 5)
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
                st.experimental_rerun()

    # Main interaction
    query = st.text_input(
        "Ask your question",
        key="user_query",
        placeholder="e.g., Why did Alice follow the White Rabbit?",
    )

    if query:
        with st.spinner("Thinking..."):
            answer, sources = get_answer(
                vectordb,
                query,
                use_reranker=use_reranker,
                final_top_n=final_top_n,
            )
        st.markdown("### Question")
        st.write(query)
        st.markdown("### Answer")
        st.write(answer)

        if show_sources and sources:
            with st.expander("Citations"):
                for d, score in sources:
                    m = d.metadata or {}
                    chunk_no = m.get('chunk_number', '?')
                    source = m.get('source', 'unknown')
                    summary = (m.get('chunk_summary') or '').strip()
                    if len(summary) > 160:
                        summary = summary[:160].rstrip() + "..."
                    st.markdown(f"- **Source**: {source} | **Chunk**: {chunk_no} | **Summary**: {summary}")

if __name__ == "__main__":
    main()