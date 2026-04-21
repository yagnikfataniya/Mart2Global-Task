import uuid
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Packages.helpers import parse_file

load_dotenv()

app = FastAPI()

# Initialize LangChain Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Chroma Vector Store via LangChain
vectorstore = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialize Chat Model for Generation
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Define the system prompt for RAG
system_prompt = (
    "You are a precise retrieval-augmented assistant. "
    "Answer only with information grounded in the retrieved context. "
    "If the context is missing, incomplete, or does not support the answer, say you do not know. "
    "Prefer the most directly relevant facts, avoid speculation, and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = prompt | llm | StrOutputParser()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    # Parse text from file
    text = parse_file(file)

    docs = text_splitter.create_documents(
        [text],
        metadatas=[{"session_id": session_id, "source": file.filename or "uploaded_file"}],
    )
    chunks = [doc.page_content for doc in docs]

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks were created from the uploaded file.")

    # Prepare metadata and IDs
    metadatas = []
    for i, doc in enumerate(docs):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = i
        metadata["total_chunks"] = len(docs)
        metadatas.append(metadata)
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]

    # Add chunks to Vector Store
    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks_indexed": len(chunks),
    }


class QueryRequest(BaseModel):
    session_id: str
    question: str


def _format_context(docs: list[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "uploaded_file")
        chunk_index = doc.metadata.get("chunk_index", "?")
        formatted_chunks.append(
            f"Source: {source} | Chunk: {chunk_index}\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def _get_session_documents(session_id: str, question: str) -> list[Document]:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.35,
            "filter": {"session_id": session_id},
        },
    )
    docs = retriever.invoke(question)
    if not docs:
        raise HTTPException(status_code=404, detail="No indexed content found for this session.")
    return docs


@app.post("/ask")
def ask(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = _get_session_documents(req.session_id, question)
    context = _format_context(docs)
    response = question_answer_chain.invoke({"input": question, "context": context})

    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "uploaded_file")
        chunk_index = doc.metadata.get("chunk_index")
        sources.append({"source": source, "chunk_index": chunk_index})

    return {
        "answer": response,
        "sources": sources,
        "chunks_used": len(docs),
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    collection = vectorstore._collection
    existing = collection.get(where={"session_id": session_id}, limit=1)
    if not existing["ids"]:
        raise HTTPException(status_code=404, detail="Session not found.")
    collection.delete(where={"session_id": session_id})
    return {"message": f"Session {session_id} deleted"}
