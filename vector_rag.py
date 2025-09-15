# vector_rag.py
import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

# ---------- CONFIG ----------
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight, good balance

# ---------- EMBEDDING WRAPPER ----------
class EmbedderWrapper:
    """Wraps SentenceTransformer for Chroma compatibility."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        return self.model.encode([text], convert_to_numpy=True)[0]

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True)

# Initialize embedding wrapper
embedding_model = EmbedderWrapper(EMBED_MODEL)

# ---------- LOAD EVENTS ----------
def load_events(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)

def create_description(event: Dict[str, Any]) -> str:
    entity = event.get("entity", "Unknown")
    eid = event.get("event_id")
    ts = event.get("timestamp")
    loc = event.get("location", "Unknown location")
    cam = event.get("camera_id", "Unknown camera")
    etype = event.get("event_type", "Unknown event")

    if entity == "Person":
        color = event.get("shirt_color", "unknown shirt color")
        return f"Event {eid}: A person (ID: {event.get('person_id')}) wearing {color} shirt appeared in {loc} at {ts}, captured by camera {cam}."
    elif entity == "Vehicle":
        vcolor = event.get("vehicle_color", "unknown color")
        plate = event.get("license_plate", "no license plate")
        return f"Event {eid}: A {vcolor} vehicle (ID: {event.get('vehicle_id')}, plate: {plate}) appeared in {loc} at {ts}, captured by camera {cam}."
    else:
        return f"Event {eid}: {etype} occurred at {loc} on {ts}, captured by {cam}."

# ---------- FLATTEN METADATA ----------
def flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert nested dicts/lists in metadata to strings."""
    flat = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            flat[k] = json.dumps(v)
        elif isinstance(v, list):
            flat[k] = str(v)
        else:
            flat[k] = v
    return flat

# ---------- BUILD VECTOR DB ----------
def build_vector_db(events_file: str):
    """Create or update Chroma DB with event descriptions."""
    events = load_events(events_file)

    texts, metadatas = [], []
    for ev in events:
        desc = ev.get("description") or create_description(ev)
        ev["description"] = desc
        texts.append(desc)
        metadatas.append(flatten_metadata(ev))  # flatten metadata

    # Initialize Chroma with embedding wrapper
    vectorstore = Chroma(
        collection_name="video_events",
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )

    # Clear existing collection if needed
    vectorstore.delete_collection()
    vectorstore = Chroma(
        collection_name="video_events",
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )

    # Add texts and persist (Chroma computes embeddings internally)
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print(f"✅ Vector DB built with {len(events)} events at {CHROMA_DIR}")

# ---------- QUERY VECTOR DB ----------
def query_vector_db(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k most relevant events for a natural query."""
    vectorstore = Chroma(
        collection_name="video_events",
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    results = vectorstore.similarity_search(query, k=k)
    return [{"description": r.page_content, **r.metadata} for r in results]

# ---------- TEST ----------
if __name__ == "__main__":
    events_file = "graph2_events.json"
    if not os.path.exists(CHROMA_DIR):
        build_vector_db(events_file)

    q = "Show me the green vehicles at the entrance"
    res = query_vector_db(q, k=3)
    for r in res:
        print(">>", r["description"])
