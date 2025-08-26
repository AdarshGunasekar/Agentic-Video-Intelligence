import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# -----------------------------
# Load mock events
# -----------------------------
with open("events.json") as f:
    EVENTS = json.load(f)

# FastAPI app
app = FastAPI()

# LLM (local Mistral via Ollama)
llm = OllamaLLM(model="mistral")

# Embedding model (Ollama embedding)
embeddings = OllamaEmbeddings(model="mistral")  # you can switch to "nomic-embed-text"

# -----------------------------
# Prepare documents for vector store
# -----------------------------
documents = []
for event in EVENTS:
    text = json.dumps(event)
    documents.append(text)

# Optional splitting (for long texts)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents(documents)

# -----------------------------
# Create Chroma Vector DB
# -----------------------------
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_store")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# API Models
# -----------------------------
class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"msg": "Agentic LLM + Vector RAG API running"}


@app.post("/query")
def query(q: Query):
    # Retrieve relevant events
    relevant_docs = retriever.get_relevant_documents(q.question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Build prompt
    prompt = f"Context:\n{context}\n\nAnswer the question: {q.question}"
    answer = llm.invoke(prompt)

    return {"question": q.question, "retrieved_events": context, "answer": answer}


@app.post("/summary")
def summary(q: Query):
    # Feed events into LLM
    context = f"Events:\n{json.dumps(EVENTS, indent=2)}"
    prompt = f"{context}\n\nProvide a concise summary based on this request: {q.question}"

    answer = llm.invoke(prompt)
    return {"question": q.question, "summary": answer}
