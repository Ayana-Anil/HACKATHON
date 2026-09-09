# Syllabus Assistant

RAG-based Q&A over an uploaded syllabus PDF. Fully local: `pypdf` for parsing,
`sentence-transformers` for embeddings, FAISS for vector search, Ollama for the LLM.

## Project Structure
```
syllabus-assistant/
├── backend/
│   ├── main.py           # FastAPI app — /upload, /ask         (Role 7)
│   ├── parser.py         # pypdf text extraction                (Role 1)
│   ├── chunker.py        # text chunking with overlap           (Role 2)
│   ├── embedder.py       # sentence-transformers wrapper        (Role 3)
│   ├── vector_store.py   # FAISS in-memory store                (Role 4)
│   ├── retriever.py      # question -> relevant chunks          (Role 5)
│   ├── llm.py            # Ollama wrapper + grounded prompt      (Role 6)
│   └── requirements.txt
├── frontend/
│   ├── index.html        # upload + chat UI                     (Role 8)
│   ├── style.css
│   └── app.js             # wires UI to backend API              (Role 9)
├── tests/
│   └── test_questions.md # QA test set + setup checklist        (Role 10)
├── data/
│   └── sample_syllabi/   # put test PDFs here
└── README.md
```

## Setup

1. **Ollama** (LLM):
   ```bash
   ollama pull llama3      # or phi3 / mistral — whatever's fastest on your machine
   ollama serve
   ```

2. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

3. **Frontend**:
   Just open `frontend/index.html` directly in a browser (no build step needed).

## Usage
1. Upload a syllabus PDF.
2. Ask questions in the chat box.
3. Answers are grounded in the syllabus text; unanswerable questions get
   "I couldn't find that in the syllabus."

## Notes
- v1 supports PDF only (via `pypdf`).
- All inference (embeddings + LLM) runs locally — no API keys, no external calls.
- State is in-memory only; restarting the backend clears uploaded docs.
