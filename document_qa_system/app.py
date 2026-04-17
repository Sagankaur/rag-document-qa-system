import streamlit as st
import os
import tempfile
import logging
from dotenv import load_dotenv
from utils import extract_text_from_pdf, extract_text_from_txt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be called early
st.set_page_config(page_title="Document QA System", page_icon="📚", layout="wide")

from rag_pipeline import MinimalRAG

# Load environment variables (API Key)
load_dotenv()

# --- Cache Resource for Model ---
# We ONLY cache the SentenceTransformer model to avoid reloading. 
# We don't cache the RAG pipeline so we can dynamically change chunk_size!
@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = get_embedding_model()

# Initialize session state for UI persistence
if "index_built" not in st.session_state: st.session_state.index_built = False
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "last_answer" not in st.session_state: st.session_state.last_answer = ""
if "last_chunks" not in st.session_state: st.session_state.last_chunks = []
if "raw_text" not in st.session_state: st.session_state.raw_text = ""
if "rag" not in st.session_state: st.session_state.rag = None

st.title("📚 Document QA System (RAG)")
st.write("Upload your PDF or TXT files, and ask questions based ONLY on their content.")

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get your key from Google AI Studio"
    )
    
    st.divider()
    st.header("🎛️ Tuning Parameters")
    st.caption("Change these to demonstrate RAG mechanics during interviews.")
    
    chunk_size = st.slider("Chunk Size (words)", min_value=100, max_value=1000, value=300, step=50, help="How much text the embedding model looks at for one chunk.")
    chunk_overlap = st.slider("Chunk Overlap (words)", min_value=0, max_value=200, value=50, step=10, help="Overlap to preserve context across chunk boundaries.")
    top_k = st.slider("Top K Retrieval", min_value=1, max_value=10, value=3, step=1, help="Number of chunks retrieved to build the prompt.")
    
    # We dynamically initialize/update RAG if it's missing or if chunk_size/overlap changed
    if st.session_state.rag is None or st.session_state.rag.chunk_size != chunk_size or st.session_state.rag.chunk_overlap != chunk_overlap:
        st.session_state.rag = MinimalRAG(embedding_model=embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.index_built = False # Force rebuild if settings change
        
    if api_key_input:
        st.session_state.rag.set_api_key(api_key_input)
    else:
        st.warning("Please enter a Gemini API Key to enable generation.")
        
    st.divider()
    
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # Quick indexing function that reads from raw text
    def build_index_from_raw():
        with st.spinner("Building vector index with new settings..."):
            success = st.session_state.rag.build_index(st.session_state.raw_text)
            if success:
                st.session_state.index_built = True
                st.session_state.last_query = ""
                st.session_state.last_answer = ""
                st.session_state.last_chunks = []
                st.success("✅ Index built successfully!")
            else:
                st.error("❌ Failed to build index.")

    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please upload at least one file.")
        else:
            with st.spinner("Extracting text from documents..."):
                combined_text = ""
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
                        temp_file.write(file.getbuffer())
                        temp_path = temp_file.name
                        
                    try:
                        if file.name.lower().endswith(".pdf"):
                            combined_text += extract_text_from_pdf(temp_path) + "\n\n"
                        elif file.name.lower().endswith(".txt"):
                            combined_text += extract_text_from_txt(temp_path) + "\n\n"
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                if combined_text.strip():
                    st.session_state.raw_text = combined_text
                    build_index_from_raw()
                else:
                    st.error("❌ No extractable text found in the uploaded documents.")

    # Show a rebuild button if text exists but index is not built (happens when sliders change)
    elif st.session_state.raw_text and not st.session_state.index_built:
        st.warning("⚠️ Chunk settings changed. You must rebuild the index.")
        if st.button("Rebuild Index with Current Settings"):
            build_index_from_raw()

# --- Main App: Q&A ---
if st.session_state.index_built:
    st.subheader("Ask a Question")
    
    with st.form(key="qa_form"):
        query = st.text_input("What would you like to know from the documents?")
        submit_button = st.form_submit_button(label="Get Answer")
        
    if submit_button:
        if not query.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.rag.llm:
            st.error("API Key not found or invalid. Please update it in the sidebar.")
        else:
            with st.spinner(f"Searching for answers... (Retrieving top {top_k} contexts)"):
                retrieved_chunks_with_scores = st.session_state.rag.retrieve(query, top_k=top_k)
                # Pass just the text strings to the LLM
                chunk_texts = [c[0] for c in retrieved_chunks_with_scores]
                answer = st.session_state.rag.generate_answer(query, chunk_texts)
                
                # Store in session state
                st.session_state.last_query = query
                st.session_state.last_answer = answer
                st.session_state.last_chunks = retrieved_chunks_with_scores

    # Display results
    if st.session_state.last_answer:
        st.write(f"**Question:** {st.session_state.last_query}")
        st.write("**Answer:**")
        st.info(st.session_state.last_answer)
        
        with st.expander(f"🔍 Show Retrieved Evidence Chunks ({len(st.session_state.last_chunks)} retrieved)"):
            st.caption("Lower L2 Distance indicates a closer semantic match to your query.")
            for i, (chunk, score) in enumerate(st.session_state.last_chunks):
                st.markdown(f"**Chunk {i+1}** | `L2 Distance: {score:.4f}`")
                st.write(chunk)
                st.divider()
else:
    if not st.session_state.raw_text:
        st.info("👈 Please upload and process documents from the sidebar before asking questions.")
