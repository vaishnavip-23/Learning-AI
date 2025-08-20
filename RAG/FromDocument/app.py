import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

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
DATA_PATH = os.path.join(os.path.dirname(__file__), "alice_in_wonderland.md")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION = "docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def check_api_key():
    """Check if API key is properly set"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment!")
        st.info("Please ensure your .env file contains: GEMINI_API_KEY=your_key_here")
        return False
    os.environ["GOOGLE_API_KEY"] = api_key
    return True

def is_meaningful_chunk(text: str) -> bool:
    """Filter out boilerplate content"""
    skip_patterns = [
        "Project Gutenberg",
        "THE MILLENNIUM FULCRUM EDITION",
        "Contents",
        "*      *      *",
        "trademark",
        "license",
        "copyright"
    ]
    return not any(pattern.lower() in text.lower() for pattern in skip_patterns)

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system"""
    try:
        # Load and preprocess document
        loader = UnstructuredMarkdownLoader(DATA_PATH, show_progress=True)
        docs = loader.load()
        
        # Configure text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True,
            strip_whitespace=True,
            add_start_index=True,
        )
        
        # Split and filter chunks
        chunks = splitter.split_documents(docs)
        filtered_chunks = [
            chunk for chunk in chunks 
            if is_meaningful_chunk(chunk.page_content) and len(chunk.page_content.strip()) > 50
        ]
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        
        # Create vector store
        vectordb = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION
        )
        
        return vectordb, embeddings
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None

def get_answer(vectordb, query: str, k: int = 3):
    """Get answer for the query"""
    try:
        # Use MMR retrieval
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k * 2,
                "fetch_k": k * 4,
                "lambda_mult": 0.7
            }
        )
        
        # Get and filter documents
        docs = retriever.get_relevant_documents(query)
        seen_content = set()
        filtered_docs = []
        
        for doc in docs:
            start_index = doc.metadata.get('start_index', None)
            if start_index is None:
                continue
            content = ' '.join(doc.page_content.split())
            if content in seen_content:
                continue
            seen_content.add(content)
            filtered_docs.append((doc, start_index))
        
        # Sort and take top 3
        filtered_docs.sort(key=lambda x: x[1])
        final_docs = [doc for doc, _ in filtered_docs[:3]]
        
        if not final_docs:
            return "No relevant information found."
        
        # Prepare context
        contexts = []
        for doc in final_docs:
            start_idx = doc.metadata.get('start_index', 0)
            context = f"[Story position {start_idx}]:\n{doc.page_content}"
            contexts.append(context)
        formatted_context = "\n\n---\n\n".join(contexts)
        
        # Create prompt
        template = """You are helping answer questions about Alice in Wonderland. Use only the provided context to answer.
If you can't find the answer in the context, say "Based on the provided context, I cannot answer this question."

Context (in story order):
{context}

Question: {question}

Instructions:
1. Use only information from the context
2. Be specific and quote relevant parts
3. Follow the story's sequence when describing events
4. If information is incomplete, say so

Answer:"""
        
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        formatted = prompt.format(context=formatted_context, question=query)
        
        # Get response from Gemini
        chat = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.2,
            top_p=0.85,
            top_k=30,
            max_output_tokens=512
        )
        response = chat.invoke(formatted)
        
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.title("🎩 Alice in Wonderland Q&A")
    st.write("Ask questions about Alice in Wonderland and get answers based on the original text!")
    
    # Check API key first
    if not check_api_key():
        return
    
    # Initialize RAG system
    with st.spinner("Initializing the system..."):
        vectordb, embeddings = initialize_rag()
        if vectordb is None:
            return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_input("Ask your question:", placeholder="e.g., What happens at the tea party?")
        
        if query:
            with st.spinner("Searching for answer..."):
                answer = get_answer(vectordb, query)
                
                # Display answer in a nice format
                st.write("### Answer")
                st.write(answer)
    
    with col2:
        # Example questions
        st.write("### Try these examples:")
        examples = [
            "What happens at the tea party?",
            "How does Alice meet the Mad Hatter?",
            "What does the Queen of Hearts say?",
            "Describe the Cheshire Cat's appearance"
        ]
        for example in examples:
            if st.button(example):
                with st.spinner("Searching for answer..."):
                    answer = get_answer(vectordb, example)
                    # Switch to col1 to display answer
                    with col1:
                        st.write("### Answer")
                        st.write(answer)

if __name__ == "__main__":
    main()