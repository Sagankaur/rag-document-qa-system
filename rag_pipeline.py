import os
import faiss
import numpy as np
import logging
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MinimalRAG:
    """A minimal Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, embedding_model=None, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # We use a simple, fast open-source model for local embeddings
        self.embedding_model = embedding_model if embedding_model else SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.llm = None
        
        # Load env key if available at init
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.set_api_key(api_key)

    def set_api_key(self, api_key: str):
        """Configure the Gemini API key safely."""
        try:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
            return True
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            self.llm = None
            return False

    def chunk_text(self, text: str) -> list[str]:
        """Splits raw text into overlapping chunks."""
        words = text.split()
        chunks = []
        if not words:
            return chunks
            
        for i in range(0, len(words), max(1, self.chunk_size - self.chunk_overlap)):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def build_index(self, text: str) -> bool:
        """Creates FAISS index from text."""
        self.chunks = self.chunk_text(text)
        if not self.chunks:
            logger.warning("No chunks generated. Text might be empty.")
            return False
            
        logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")
        try:
            embeddings = self.embedding_model.encode(self.chunks)
            # Ensure 2D float32 array for faiss
            embeddings_np = np.array(embeddings).astype('float32')
            
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
            return True
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
        
    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Retrieves top_k chunks relevant to the query along with their L2 distance scores."""
        if not self.index or not self.chunks:
            return []
            
        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
            
            results = []
            # faiss returns a matrix of indices and distances.
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.chunks):
                    results.append((self.chunks[idx], float(dist)))
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Generates an answer using Gemini, grounded in context."""
        if not self.llm:
            return "Error: Gemini API key is not configured or is invalid. Please set it in the sidebar."
            
        if not context_chunks:
            return "I could not find relevant information in the uploaded documents."
            
        # Grounded prompt template
        context_str = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""
You are a helpful AI assistant tasked with answering user questions based ONLY on the provided document excerpts.
Follow these rules strictly:
1. Do not invent facts. 
2. If the answer is not present in the excerpts, say EXACTLY: "I could not find that in the uploaded documents."
3. Cite the retrieved chunk numbers in your answer where appropriate (e.g., "[Chunk 1]").

Document Excerpts:
{context_str}

User Question: {query}

Answer:"""
        
        try:
            response = self.llm.generate_content(prompt)
            # Gemini block check
            if not response.parts:
                 return "The model declined to answer. This may be due to safety filters."
            return response.text
        except Exception as e:
            logger.error(f"API Generation Error: {e}")
            return f"Error communicating with Gemini: {str(e)}"
