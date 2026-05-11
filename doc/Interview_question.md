### 🎤 1. Project Explanation (The "Elevator Pitch")

**What it is:** 
"I built a local Retrieval-Augmented Generation (RAG) system that allows users to securely query proprietary documents—like PDFs and Text files—without risking data leakage to cloud providers or suffering from AI hallucinations."

**How it works:**
"I built the backend using pure Python. When a document is uploaded, the system extracts the text using `pypdf`, chunking the words with overlap to maintain context. I generate vector embeddings for each chunk completely locally using a HuggingFace `SentenceTransformer` model, and index those vectors into memory using Facebook’s `FAISS` library. 

When a user asks a question via the Streamlit frontend, the system calculates the L2 mathematical distance to find the top most relevant chunks, wraps them in a strict zero-shot constraint prompt, and feeds them to the Gemini API to synthesize the final answer. To build trust, I engineered a transparency UI that forces the app to show exactly which chunks of text it used and their similarity scores."

---

### ❓ 2. Likely Interview Questions & Answers

**Q: Why did you choose FAISS over a managed vector database like Pinecone or Weaviate?**
**A:** "Because my architecture goal was low-latency and local-first. Pinecone is fantastic for massive, persistent datasets shared across distributed systems. However, for a session-based document QA tool, making network hops to a cloud DB for querying 100 PDF chunks is massive overkill. FAISS runs directly in RAM via C++ bindings, making retrieval virtually instantaneous for this scale."

**Q: If I upload a 500-page PDF, how does your system handle it? What are the bottlenecks?**
**A:** "The immediate bottleneck is the embedding generation. Running `all-MiniLM-L6-v2` on CPU to encode thousands of chunks will cause heavy freezing. To fix this, I would need to implement background batch-processing (e.g., using Celery or FastAPI background tasks) instead of doing it synchronously in Streamlit, and ideally offload the tensor math to a GPU."

**Q: How do you mathematically measure which chunks are 'relevant'?**
**A:** "FAISS calculates the **L2 Distance** (Euclidean distance) between the vector of the user's question and the vectors of the document chunks. The lower the distance, the more semantically similar the concepts are in the high-dimensional space."

**Q: Why use 'Chunk Overlap'? What happens if overlap is 0?**
**A:** "If overlap is 0, we risk slicing a sentence structurally in half. For instance, chunk 1 might end with 'The company's revenue...' and chunk 2 starts with '...increased by 40%'. The embedding model evaluates those separately, meaning neither chunk understands the full context, leading to retrieval failure."

---

### 🐛 3. The Debugging Story (Use this for "Tell me about a time you solved a hard bug")

"While building the UI in Streamlit, I hit a massive roadblock regarding stateless web applications. After a user got their answer, they could click an 'Expander' toggle to see the source evidence. But the moment they clicked the toggle, the entire answer disappeared. 

I had to dive deep into Streamlit's architecture to realize that clicking the toggle triggered a top-to-bottom script rerun. Because the 'Get Answer' button state reverted to `False` on the rerun, the application wiped the generated text. 

I solved this by dropping my reliance on standard python variables and engineering a robust **Session State caching system**. I hijacked Streamlit's memory dictionary to explicitly trap the `last_query`, `last_answer`, and `last_retrieved_chunks`. It taught me a vital lesson about state-management in reactive frontends, and the final app became incredibly smooth and bug-free."

---

### ⚖️ 4. Engineering Trade-Offs

Be prepared to explain *why* you built it this way, recognizing that every choice has a cost:

1. **Local Embeddings vs. API Embeddings (e.g., OpenAI `text-embedding-3-small`)**
   * *Tradeoff:* Local sentence-transformers cost nothing and guarantee privacy. However, they struggle heavily with non-English languages and highly complex reasoning compared to OpenAI's massive proprietary embedding models.
2. **Naive Word-Chunking vs. Semantic Splitting**
   * *Tradeoff:* Splitting by a math rule (e.g., every 300 words) is fast (O(N) operation) and easy to code. However, semantic splitting (using NLP libraries like NLTK to carefully only split at period marks) is much better for contexts but computationally slower and harder to maintain.
3. **Streamlit UI vs. React/FastAPI Stack**
   * *Tradeoff:* Streamlit allowed me to deploy a functional, data-heavy dashboard in 100 lines of code to prove the RAG concept. But it is difficult to deploy at scale and lacks the granular API endpoint separation a true React + FastAPI backend would afford.

---

### 🚀 5. Roadmap for Improvements (Show them you know what happens in Production)

If asked "how would you improve this for enterprise?", mention these:
* **Re-Ranking / Cross-Encoding**: "Right now, I take the top 3 FAISS results. In production, I would retrieve the top 20, and pass them through a 'Cross-Encoder/Reranker' model (like Cohere) which is much slower but highly accurate, to order the top 3 perfectly."
* **OCR Integration**: "PyPDF essentially ignores images of text. I would integrate `Tesseract` or `LlamaParse` to handle scanned documents."
* **Hybrid Search (Ensembling)**: "Vector search struggles with exact keyword matching (like searching for specific Employee ID numbers). I would combine vector search with BM25 sparse keyword search for a hybrid approach."

---

### 📄 6. Resume Bullets

*Pick 2 or 3 of these based on the job description you are applying for:*

**For a backend/ML role:**
* Engineered an end-to-end local Retrieval-Augmented Generation (RAG) system utilizing Python, successfully synthesizing query answers from proprietary documents with zero forced-hallucination.
* Optimized vector similarity search latency by indexing local CPU-bound semantic embeddings via the FAISS flat L2 algorithm, bypassing cloud database network bottlenecks.
* Curated dynamic text assimilation by engineering overlapping NLP chunking logic, maximizing contextual preservation and mitigating vector distortion across page boundaries.

**For a full-stack / AI integration role:**
* Architected a fully interactive Document QA web-dashboard utilizing Streamlit and session-state memory management, orchestrating seamless PDF ingestion to LLM generation pipelines.
* Integrated the Google Gemini LLM via explicit constrained zero-shot prompt engineering, systematically citing local L2 distance scores and forcing refusal conditions for ungrounded queries.
* Developed self-contained `tempfile` memory-management fail-safes during I/O buffer operations, drastically reducing orphan-file disk consumption on application crashes.