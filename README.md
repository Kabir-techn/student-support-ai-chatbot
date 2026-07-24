# 🎓 AI Student Support Services Chatbot

A production-ready, Retrieval-Augmented Generation (RAG) chatbot that answers
student questions about admissions, fees, scholarships, hostel, library,
placements, exams, attendance, academic calendar, and more — grounded in
official college documents, with source citations and a confidence score on
every answer.



---

## ✨ Features

- **Conversational AI** with follow-up / context awareness ("What is hostel
  fee?" → "Does that include mess?")
- **RAG pipeline**: PDF/DOCX/TXT/CSV → chunking → Sentence-Transformer
  embeddings → FAISS similarity search → grounded LLM answer
- **FAQ Mode**: exact + semantic-similarity cache for instant answers to
  common questions
- **Source citation** on every RAG answer (filename + page number)
- **Confidence score** on every answer; low-confidence questions are
  refused rather than hallucinated
- **Conversation memory** per session
- **Chat history**, **👍/👎 feedback**, and **analytics** persisted to SQLite
- **Admin dashboard**: upload/delete knowledge-base documents, rebuild the
  vector index, view analytics & common questions, export logs as CSV
- **Pluggable LLM provider**: OpenAI, Ollama (local LLM), or a zero-dependency
  offline extractive fallback — the app works with **no API key at all**
- Dark mode, typing/loading states, chat bubbles, suggested questions

---

## 🏗️ Architecture

```
Documents (PDF/DOCX/TXT/CSV)
        │
        ▼
   Chunking (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
   Embeddings (Sentence-Transformers, all-MiniLM-L6-v2)
        │
        ▼
   FAISS Vector Index  ◄────────────┐
        │                            │  Admin: upload / rebuild
        ▼                            │
   Retriever (top-K, cosine sim) ────┘
        │
        ▼
   Confidence scoring ──► below threshold ──► "Not confident" fallback
        │
        ▼
   LLM (OpenAI / Ollama / local extractive) + conversation memory
        │
        ▼
   Cited, confidence-scored answer ──► SQLite (chat_history, feedback)
```

### Folder structure

```
student_support_ai/
├── app.py                     # FastAPI entry point
├── config.py                  # Centralized settings (env-driven)
├── requirements.txt
├── Dockerfile                 # Backend container
├── Dockerfile.frontend        # Streamlit container
├── docker-compose.yml
├── pytest.ini
│
├── frontend/
│   └── streamlit_app.py       # Streamlit UI (chat + admin)
│
├── backend/
│   ├── chatbot.py             # Orchestrator: FAQ -> RAG -> memory -> DB
│   ├── rag.py                 # Retrieval + LLM generation + confidence
│   ├── embeddings.py          # Sentence-Transformers + FAISS wrapper
│   ├── document_loader.py     # PDF/DOCX/TXT/CSV loading + chunking
│   ├── memory.py              # Short-term conversation memory
│   ├── prompt.py              # LLM prompt templates
│   ├── faq.py                 # FAQ cache + semantic matching
│   ├── database.py            # SQLAlchemy models + SQLite persistence
│   └── routes.py               # FastAPI route handlers
│
├── database/                  # student.db (SQLite, created at runtime)
├── documents/                 # Knowledge base (sample PDFs + CSV included)
├── vectorstore/                # FAISS index + metadata (built at runtime)
├── models/                    # Reserved for any locally-cached models
├── logs/                      # Rotating app logs
├── utils/
│   ├── logger.py
│   └── build_index.py         # CLI: `python -m utils.build_index`
└── tests/                     # pytest unit tests (36 tests, offline-friendly)
```

---

## 🚀 Quick Start (local, no Docker)

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — works out of the box in "local" mode)
cp .env.example .env

# 4. Build the vector index + seed the database (first run only;
#    also happens automatically on first chat request)
python -m utils.build_index

# 5. Start the backend API
uvicorn app:app --reload --port 8000

# 6. In a second terminal, start the frontend
streamlit run frontend/streamlit_app.py
```

Open the Streamlit UI (usually `http://localhost:8501`) and the API docs at
`http://localhost:8000/docs`.

### Enabling a real LLM (optional)

By default `LLM_PROVIDER=local`, which returns a grounded excerpt from the
retrieved document instead of calling an external LLM — fully offline, zero
cost, zero hallucination risk. To get more natural, synthesized answers:

**OpenAI:**
```bash
# in .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

**Ollama (self-hosted, free):**
```bash
# install & run Ollama, then pull a model
ollama pull llama3

# in .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT_SECONDS=90
```

Verify Ollama is reachable before starting the app:
```bash
curl http://localhost:11434/api/tags   # should list your installed models
```

If the chatbot silently falls back to raw excerpts even with `LLM_PROVIDER=ollama`
set, check the backend logs (`logs/app.log` or the terminal running `uvicorn`)
for a line like `LLM generation failed (...); using extractive fallback` — the
error message tells you exactly what went wrong:
- *"Could not reach Ollama"* → Ollama isn't running; start it with `ollama serve`
  or open the Ollama desktop app.
- *"model not found"* → `OLLAMA_MODEL` in `.env` doesn't match an installed
  model; run `ollama list` to see exact names, or `ollama pull <name>`.
- *"did not respond within Ns"* → the model is still loading into memory on
  first use (common on CPU); try the question again, or raise
  `OLLAMA_TIMEOUT_SECONDS`.

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

This starts:
- `backend` — FastAPI on port `8000`
- `frontend` — Streamlit on port `8501`, talking to the backend over the
  compose network

Knowledge-base documents, the vector index, the SQLite database, and logs are
mounted as volumes so they persist across container restarts.

---

## 🔌 API Reference (key endpoints)

| Method | Endpoint                        | Description                              |
|--------|----------------------------------|-------------------------------------------|
| GET    | `/health`                        | Liveness check                            |
| POST   | `/chat`                          | Send a message, get a cited answer        |
| GET    | `/chat/history/{session_id}`     | Retrieve a session's chat history         |
| POST   | `/chat/feedback`                 | Submit 👍/👎 feedback on an answer         |
| GET    | `/chat/suggested-questions`      | Get sample questions for the UI           |
| POST   | `/admin/upload`                  | Upload a new PDF/DOCX/TXT/CSV document    |
| GET    | `/admin/documents`               | List knowledge-base documents             |
| DELETE | `/admin/documents/{filename}`    | Remove a document                         |
| POST   | `/admin/rebuild-index`           | Re-embed all documents, rebuild FAISS     |
| GET    | `/admin/analytics`               | Message/session/feedback counts           |
| GET    | `/admin/common-questions`        | Most frequently asked questions           |
| GET    | `/admin/export-logs`             | Download full chat history as CSV         |

Full interactive docs (Swagger UI) are auto-generated at `/docs` when the
backend is running.

---

## 🧪 Testing

```bash
pytest -v
```

36 unit tests cover document loading (PDF/DOCX/TXT/CSV), chunking, FAISS
build/search/persistence, confidence scoring, source citation, FAQ matching,
conversation memory, SQLite persistence, and every FastAPI route. Tests use a
deterministic fake embedding function (see `tests/conftest.py`) so the full
suite runs in seconds, fully offline — no model download or network access
required in CI.

---

## 🧠 Responsible AI / Hallucination Reduction

- Every RAG answer is **grounded only in retrieved document chunks** — the
  prompt explicitly instructs the LLM not to use outside knowledge.
- A transparent **confidence score** (top retrieval cosine similarity) gates
  every response. Below `CONFIDENCE_THRESHOLD` (default `0.45`), the bot
  declines to answer and points the student to Student Support instead of
  guessing.
- **Source citations** (filename + page) let students verify any answer
  against the original document.
- The `local` LLM provider mode returns a direct excerpt with no generation
  step at all, for a zero-hallucination-risk deployment option.

---

## ⚙️ Configuration

All settings live in `config.py` and are overridable via environment
variables / `.env` (see `.env.example`). Notable ones:

| Variable                | Default   | Description                                  |
|--------------------------|-----------|-----------------------------------------------|
| `LLM_PROVIDER`            | `local`   | `openai` \| `ollama` \| `local`               |
| `EMBEDDING_MODEL`         | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K`                   | `4`       | Chunks retrieved per query                    |
| `CONFIDENCE_THRESHOLD`    | `0.45`    | Minimum similarity to answer (else fallback)  |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Text splitting parameters           |
| `MAX_HISTORY_TURNS`       | `8`       | Conversation turns kept in memory             |

---

## 📚 Knowledge Base

Sample documents are included in `documents/`:
- `fee_structure.pdf`
- `academic_calendar.pdf`
- `scholarship.pdf`
- `placements.csv`

Add your own college's PDFs/DOCX/TXT/CSV files to this folder (or via the
Admin → Documents tab), then click **Rebuild FAISS Vector Database** to index
them.

---

## 🛣️ Extending This Project

- Swap SQLite for Postgres by changing `DATABASE_URL` in `.env` — the
  SQLAlchemy layer needs no other changes.
- Add authentication (e.g. student login) by wrapping the FastAPI routes with
  a dependency that validates a JWT/session.
- Swap the embedding model for a multilingual one (e.g.
  `paraphrase-multilingual-MiniLM-L12-v2`) to support regional languages.
- Add a proper LangChain `ConversationalRetrievalChain` if you want built-in
  question-reformulation instead of the lightweight version in `prompt.py`.

---

## 📄 License

Provided as-is for educational / capstone use.
