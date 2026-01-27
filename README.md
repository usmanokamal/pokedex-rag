# Pokemon RAG Chatbot

Simple RAG-based chatbot over `data/Pokemon.csv` with FAISS + FastAPI and Gemini for answers.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` from `.env.example` and add your Gemini key when ready.

## Run

```bash
uvicorn app:app --reload
```

The first run builds the FAISS index into `storage/`.

## API

- `GET /health` -> basic health check
- `POST /chat`

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "pokemon with 80 hp", "top_k": 5}'
```

If `GEMINI_API_KEY` is not set, the API returns a formatted list from the retrieved rows.

## Notes

- Change embeddings model with `LOCAL_EMBEDDING_MODEL`.
- If you change the dataset or embedding model, delete `storage/` to rebuild the index.
