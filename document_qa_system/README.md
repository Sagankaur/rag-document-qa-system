# 📚 Document QA System (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDF and TXT documents, and ask questions based **strictly** on the content of those documents. 

Built as a clean, local-first mini-project for AI engineers and interns to showcase fundamental GenAI engineering concepts.

## 🌟 Features
- **Document Upload**: Supports uploading multiple `.txt` or `.pdf` files.
- **Local Chunking & Embeddings**: Uses open-source `sentence-transformers` and `FAISS` to generate embeddings and retrieve chunks efficiently locally. No vector database cloud setup required.
- **Grounded Generation**: Uses Gemini 1.5 Flash to generate answers grounded **only** in the retrieved document context. It actively avoids hallucination by citing chunk numbers and declining to answer if the context lacks the information.
- **Explainability**: A built-in expander UI element reveals the exact text chunks the LLM retrieved and used to generate its answer.

## 🏗️ Architecture

1. **Extraction**: `PyPDF2` (for PDF) and standard Python I/O (for TXT) extract raw text.
2. **Chunking**: Text is split into fixed-size overlapping text chunks (e.g., 300 words, 50 word overlap) to preserve context.
3. **Embedding**: `all-MiniLM-L6-v2` (SentenceTransformer) encodes chunks into high-dimensional vectors locally.
4. **Vector Store**: `FAISS` creates a local in-memory L2 index of the chunk embeddings.
5. **Retrieval**: User queries are embedded, and FAISS retrieves the top-K nearest neighbors (the most relevant chunks).
6. **Generation**: The Gemini API receives a strict zero-shot prompt wrapped around the extracted chunks to generate the final answer.

## 🚀 Setup Instructions

1. **Prerequisites**: Python 3.9+ 
2. **Clone the repository** (or download the files).
3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
4. **Environment Variables**:
Rename `.env.example` to `.env` and put in your Google Gemini API key:
```env
GEMINI_API_KEY="your_api_key_here"
```
*(You can get a free Gemini key from Google AI Studio).*

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

1. Open the local URL provided by Streamlit in your browser.
2. Upload your documents in the left sidebar.
3. Click **"Process Documents"**.
4. Type your question in the main text box.
5. Expand the "Show Retrieved Evidence Chunks" section to see the source information!

## 🚧 Limitations & Future Improvements
* **In-memory FAISS**: Currently, the index is lost upon restarting the server. A future improvement would be saving the `.index` file to disk for persistence.
* **Naive Chunking**: Uses standard word-based chunking. Could be upgraded to semantic chunking or LangChain's RecursiveCharacterTextSplitter.
* **PDF Parse Quality**: PyPDF2 struggles with complex tables/layouts. Pymupdf or OCR could be integrated for better extraction accuracy.
