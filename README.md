# Minimal RAG API

FastAPI service for upload-and-query retrieval augmented generation over user-provided documents. The app ingests supported files, chunks them, stores embeddings in Chroma, and answers questions against the uploaded session only.

## Features

- Upload a document and create an isolated `session_id`
- Parse `PDF`, `CSV`, `XLS/XLSX`, `TXT`, and `MD` files
- Chunk content and index it into a local Chroma database
- Ask grounded questions against one uploaded session
- Return answer metadata including retrieved source chunks
- Delete a session and remove its indexed chunks

## Requirements

- Python 3.11+
- An OpenAI API key

## Setup

1. Create and activate a virtual environment.
2. Install the project dependencies.
3. Add your OpenAI API key to `.env`.
4. Start the FastAPI server.

Example `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

Example install and run flow:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install fastapi uvicorn python-dotenv pypdf pandas openpyxl chromadb langchain langchain-openai langchain-chroma langchain-text-splitters
uvicorn app:app --reload
```

## API

### `POST /upload`

Uploads and indexes a file.

Form data:

- `file`: document file

Example response:

```json
{
  "session_id": "f1f7d6c2-3f77-4d37-8c1e-5f4d8654d3d4",
  "filename": "sample.pdf",
  "chunks_indexed": 8
}
```

### `POST /ask`

Queries indexed content for one session.

Request body:

```json
{
  "session_id": "f1f7d6c2-3f77-4d37-8c1e-5f4d8654d3d4",
  "question": "What are the main conclusions?"
}
```

Example response:

```json
{
  "answer": "The document says ...",
  "sources": [
    {
      "source": "sample.pdf",
      "chunk_index": 2
    }
  ],
  "chunks_used": 6
}
```

### `DELETE /session/{session_id}`

Deletes all indexed chunks for a session.

## Project Structure

```text
.
|-- app.py
|-- Packages/
|   `-- helpers.py
|-- chroma_db/
`-- .env
```

## Notes

- Chroma data is stored locally in `chroma_db/`.
- The app currently initializes embeddings and the chat model at import time.
- If a file has no readable text, the upload is rejected with an HTTP `400`.
