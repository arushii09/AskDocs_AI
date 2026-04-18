# Askdocs AI — AI-Powered Document Assistant

A conversational AI app that lets you upload any PDF and ask questions about it in natural language. Built using RAG (Retrieval Augmented Generation) — meaning the AI answers strictly from your document, not from general knowledge.


## ⚙️ How It Works
```
Upload PDF
    ↓
Split into chunks (LangChain)
    ↓
Convert to embeddings (HuggingFace - all-MiniLM-L6-v2)
    ↓
Store in vector database (ChromaDB)
    ↓
User asks a question
    ↓
Search ChromaDB for relevant chunks
    ↓
Send chunks + question to LLM (Groq - LLaMA 3.3)
    ↓
Get answer based on your PDF
```

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **LangChain** | AI pipeline and RAG orchestration |
| **Groq + LLaMA 3.3** | LLM for generating answers |
| **ChromaDB** | Vector database for storing embeddings |
| **HuggingFace** | Embedding model (all-MiniLM-L6-v2) |
| **Streamlit** | Web interface |
| **PyPDF** | PDF loading and parsing |

## 🚀 Run Locally

**1. Clone the repo**
**2. Install dependencies: `pip install -r requirements.txt`**
**3. Add your `GROQ_API_KEY` in a `.env` file**
**4. Run: `streamlit run app.py**

Open your browser at `http://localhost:8501`

## 📋 Features
- Upload any text-based PDF
- Ask questions in natural language
- Conversational memory — remembers previous questions in the session
- Answers strictly from your document
- Fast responses via Groq's inference API

## ⚠️ Limitations
- Works best with text-based PDFs (not scanned/image PDFs)
- Chat history resets when you upload a new PDF
- Large PDFs may take longer to index on first upload


Built by Arushi 
