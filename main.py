# main.py
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from graph_client import GraphClient
from langchain_ollama import OllamaLLM

# --- Load events (mock file) ---
with open("graph2_events.json") as f:
    EVENTS = json.load(f)

app = FastAPI(title="Agentic LLM + Graph RAG")

# --- LLM (used for /ask) ---
llm = OllamaLLM(model="mistral")

# --- Graph client ---
graph = GraphClient()
graph.ensure_constraints()

# -------- Pydantic models --------
class CypherQuery(BaseModel):
    cypher: str
    params: Optional[Dict[str, Any]] = None

class Ask(BaseModel):
    question: str

@app.on_event("shutdown")
def on_shutdown():
    graph.close()

@app.get("/")
def home():
    return {"msg": "Graph RAG API running"}

# -------- Graph Endpoints --------
@app.post("/graph/ingest")
def graph_ingest():
    graph.ingest_events(EVENTS)
    return {"status": "ok", "ingested": len(EVENTS)}

@app.post("/graph/query")
def graph_query(q: CypherQuery):
    rows = graph.run_cypher(q.cypher, q.params)
    return {"rows": rows, "count": len(rows)}

@app.get("/graph/person/{person_id}/trail")
def person_trail(person_id: str, start: Optional[str] = None, end: Optional[str] = None):
    return {"person_id": person_id, "trail": graph.person_trail(person_id, start, end)}

@app.get("/graph/vehicle/{vehicle_id}/trail")
def vehicle_trail(vehicle_id: str, start: Optional[str] = None, end: Optional[str] = None):
    return {"vehicle_id": vehicle_id, "trail": graph.vehicle_trail(vehicle_id, start, end)}

# -------- /ask Endpoint (NLQ → Cypher → reuse endpoints) --------
@app.post("/ask")
def ask(q: Ask):
    """
    Converts natural language questions into Cypher queries using few-shot examples,
    executes the exact query in Neo4j, and summarizes the results.
    """
    import json
    import re

    # --- Few-shot examples for NLQ → Cypher ---
    few_shot_examples = """
Example 1:
NLQ: "Show me all events where person P123 was involved, and also list the vehicles they used in those events."
Cypher:
MATCH (p:Person {person_id:'P123'})-[:APPEARED_IN]->(e:Event)
OPTIONAL MATCH (v:Vehicle)-[:APPEARED_IN]->(e)
OPTIONAL MATCH (e)-[:AT]->(l:Location)
OPTIONAL MATCH (e)-[:CAPTURED_BY]->(c:Camera)
WITH e, l, c,
     toString(e.timestamp) AS timestamp,
     collect(DISTINCT v.license_plate) AS vehicles
RETURN e.event_id AS event_id,
       timestamp,
       l.name AS location,
       c.camera_id AS camera_id,
       e.video_file AS video_file,
       e.frame_number AS frame_number,
       e.video_time AS video_time,
       vehicles
ORDER BY timestamp;

Example 2:
NLQ: "List all events involving vehicle V456 after January 1, 2023."
Cypher:
MATCH (v:Vehicle {vehicle_id:'V456'})-[:APPEARED_IN]->(e:Event)
WHERE e.timestamp >= datetime('2023-01-01T00:00:00')
OPTIONAL MATCH (e)-[:AT]->(l:Location)
OPTIONAL MATCH (e)-[:CAPTURED_BY]->(c:Camera)
WITH e, l, c, v,
     toString(e.timestamp) AS timestamp
RETURN e.event_id AS event_id,
       timestamp,
       l.name AS location,
       c.camera_id AS camera_id,
       e.video_file AS video_file,
       e.frame_number AS frame_number,
       e.video_time AS video_time,
       v.license_plate AS license_plate
ORDER BY timestamp;

Example 3:
NLQ: "Which person has participated in the highest number of events?"
Cypher:
MATCH (p:Person)-[:APPEARED_IN]->(e:Event)
WITH p, collect(DISTINCT e) AS events
RETURN p.person_id AS person_id,
       size(events) AS event_count,
       [ev IN events | ev.event_id] AS event_ids,
       [ev IN events | toString(ev.timestamp)] AS timestamps
ORDER BY event_count DESC
LIMIT 1;

Example 4:
NLQ: "Show all vehicles that were used by both person P123 and person P456."
Cypher:
MATCH (p1:Person {person_id:'P123'})-[:APPEARED_IN]->(e1:Event)<-[:APPEARED_IN]-(v:Vehicle)
MATCH (p2:Person {person_id:'P456'})-[:APPEARED_IN]->(e2:Event)<-[:APPEARED_IN]-(v)
WITH v, collect(DISTINCT e1) + collect(DISTINCT e2) AS events
RETURN v.license_plate AS vehicle_id,
       [ev IN events | ev.event_id] AS related_events,
       [ev IN events | toString(ev.timestamp)] AS timestamps;

Example 5:
NLQ: "Summarize the activities of person P123 across all locations."
Cypher:
MATCH (p:Person {person_id:'P123'})-[:APPEARED_IN]->(e:Event)
OPTIONAL MATCH (e)-[:AT]->(l:Location)
OPTIONAL MATCH (e)-[:CAPTURED_BY]->(c:Camera)
WITH l, collect(DISTINCT e) AS events, collect(DISTINCT c) AS cameras
RETURN l.name AS location,
       size(events) AS event_count,
       [ev IN events | ev.event_id] AS event_ids,
       [ev IN events | toString(ev.timestamp)] AS timestamps,
       [cam IN cameras | cam.camera_id] AS camera_ids,
       [ev IN events | ev.video_file] AS video_files
ORDER BY location;

Example 6:
NLQ: "Which people have traveled together in the same vehicle in at least 3 different events?"
Cypher:
MATCH (p1:Person)-[:APPEARED_IN]->(e:Event)<-[:APPEARED_IN]-(v:Vehicle)<-[:APPEARED_IN]-(p2:Person)
WHERE p1.person_id < p2.person_id
WITH p1, p2, v, collect(DISTINCT e) AS shared_events
WHERE size(shared_events) >= 3
RETURN p1.person_id AS person1,
       p2.person_id AS person2,
       v.license_plate AS vehicle_id,
       size(shared_events) AS shared_event_count,
       [ev IN shared_events | ev.event_id] AS event_ids,
       [ev IN shared_events | toString(ev.timestamp)] AS timestamps
ORDER BY shared_event_count DESC;

Example 7:
NLQ: "Who appeared in Cam2?"
Cypher: MATCH (e:Event)-[:CAPTURED_BY]->(c:Camera {camera_id:"Cam2"})
        <-[:APPEARED_IN]-(p:Person) RETURN DISTINCT p.person_id;

        
Example 8:
NLQ:  "What activities happened in the Hallway?"
Cypher: MATCH (e:Event)-[:AT]->(l:Location {name:"Hallway"})
        RETURN e.activity;
"""


    # --- Graph schema ---
    schema = """
Graph schema:
- (Person {person_id, shirt_color})
- (Vehicle {vehicle_id, type, license_plate})
- (Camera {camera_id})
- (Location {name})
- (Event {event_id, timestamp, event_type, video_file, frame_number, video_time, confidence, shirt_color, license_plate})

Relationships:
- (Person)-[:APPEARED_IN]->(Event)
- (Vehicle)-[:APPEARED_IN]->(Event)
- (Event)-[:AT]->(Location)
- (Event)-[:CAPTURED_BY]->(Camera)
- (prev:Event)-[:FOLLOWS]->(next:Event)

Rules:
- Do not invent properties. 
- Always respect relationship directions.
"""

    # --- Step 1: Prepare prompt for LLM ---
    prompt = f"""
You are an expert Cypher generator. ONLY return valid Cypher, no explanations.
Use this graph schema:


{schema}
Important:

Do not use OR between graph patterns.

If multiple alternative paths need to be checked, use either multiple OPTIONAL MATCH statements or separate queries combined with UNION.

Always return valid Cypher syntax with no placeholders, comments, or natural language.

Here are examples of NLQ → Cypher:


{few_shot_examples}

Now generate Cypher for this user question:
\"\"\"{q.question}\"\"\"
"""

    # --- Step 2: Generate Cypher ---
    cypher = llm.invoke(prompt).strip().replace("```cypher", "").replace("```", "").strip()

    # --- Step 3: Execute the exact Cypher in Neo4j ---
    try:
        result = graph.run_cypher(cypher)
    except Exception as e:
        return {"error": f"Cypher execution failed: {e}", "raw_cypher": cypher}

    # --- Step 4: Summarize results using LLM ---
    summary_prompt = f"""
You are given the Cypher query results as structured JSON.
User Question:{q.question}
Results: {json.dumps(result, indent=2)} 
Write a concise natural language summary using the exact values from the results and understand what the question is about and kindly use the insights while summarizing.
- Always mention the actual person_ids, event_ids, locations, camera_ids, video_files, timestamps, and vehicles.
- Never invent values.
- Use the exact values from the results.
- Do NOT replace event IDs with person IDs.
- Mention timestamps, cameras, and video files if available.

Summary:
"""
    summary = llm.invoke(summary_prompt).strip()

    return {
        "cypher": cypher,
        "results": result,
        "summary": summary
    }













