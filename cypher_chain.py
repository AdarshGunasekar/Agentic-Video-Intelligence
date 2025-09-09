import re
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import traceback

# ---------- CONFIG ----------
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "adhu@2580"

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
llm = OllamaLLM(model="mistral")

# ---------- PROMPTS ----------
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a valid Cypher statement to query a Neo4j graph database.

Schema:
{schema}

Instructions:
- Only use the relationship types, labels, and properties provided in the schema.
- Vehicle identifiers:
- If the user refers to something like "V###" (example: V200, V101), 
  always match using vehicle_id:
      MATCH (v:Vehicle {{vehicle_id: 'V200'}})

- If the user refers to a real license plate format (example: "KA-63-7309"),
  always match using license_plate:
      MATCH (v:Vehicle {{license_plate: 'KA-63-7309'}})
  *Never confuse the two.
- Do not confuse vehicle_id with license_plate — they are different properties.
- When a query involves colors (e.g., "red", "green", etc.):
  * Always check both shirt_color (Person-related events) and vehicle_color (Vehicle-related events).
  # Person shirt color:
# - Shirt color is stored in the Event node, not the Person node.
# - Always filter using e.shirt_color, not p.shirt_color.
# Example:
#   MATCH (p:Person)-[:APPEARED_IN]->(e:Event)
#   WHERE toLower(e.shirt_color) = toLower('red')
#   RETURN DISTINCT p.person_id;
- For vehicle color, always use `e.vehicle_color` from the `Event` node (not from Vehicle).
- For event sequences, order by `e.frame_number` if available, otherwise `e.timestamp`.
- Do not use SQL-style keywords like GROUP BY, HAVING, INSERT, UPDATE, DELETE.
- In Cypher, always use WITH for grouping/aggregation (never GROUP BY).
- Do not put graph patterns (()-[]-()) inside WHERE clauses; always use MATCH for patterns.
- Always alias aggregated values or formatted values with AS.
- If you alias a property (e.g., toString(e.timestamp) AS timestamp), 
  then always use ONLY the alias in RETURN, ORDER BY, and WHERE clauses. Do not mix raw and alias together.
- Do not return the same property both raw and aliased.
- Output only raw Cypher (no explanations, no 'Cypher:' prefix, no natural language).
- Use double curly braces {{ }} when referencing string properties inside patterns.
*Never use `OR` between patterns inside MATCH.  
*Instead:
- Use multiple MATCH clauses if both must exist.
- Use OPTIONAL MATCH if one may not exist.
- Use UNION if you want to combine results from two different patterns.
*Use **valid Cypher syntax only**.
   - Relationships: always use `()-[:REL]->()` or `()<-[:REL]-()`. 
   - Do not generate `(a)-[:REL]<-(b)` inside OPTIONAL MATCH without proper structure.
*Never include markdown fences (no ```cypher).
*Do not include SQL-like constructs (no ternary `? :` operators).
   - Use `coalesce(field1, field2)` instead.
*Avoid OR conditions inside a single MATCH.  
   - Split them into multiple MATCH + UNION if necessary.
*Always filter using `WHERE` after MATCH.
*If query references:
   - **person_id** → replace with `$person_id`.
   - **vehicle_id** → replace with `$vehicle_id`.
   - Always use property `license_plate` (not license_plate_number).
   - Allow case-insensitive search with `toLower(e.license_plate) = toLower('XYZ')`..
- Each MATCH must have a RETURN.
- The RETURN columns must match in both parts of the UNION.
- UNION automatically removes duplicates; if you want to keep them, use UNION ALL.
- Only use properties and relationships from the schema.
2. Do not invent properties that are not listed.
3. For location-based queries:
   - Remember that Event nodes do NOT have a 'location' property.
   - Instead, use (:Event)-[:AT]->(:Location {{name:'...'}}) to filter by location.
4. For person or vehicle appearances:
   - (:Person)-[:APPEARED_IN]->(:Event)
   - (:Vehicle)-[:APPEARED_IN]->(:Event)



Examples:

Question: "Show me all events where person P123 was involved."
MATCH (p:Person {{person_id:'P123'}})-[:APPEARED_IN]->(e:Event)
RETURN e.event_id AS event_id, toString(e.timestamp) AS timestamp;

Question: "Which person has participated in the highest number of events?"
MATCH (p:Person)-[:APPEARED_IN]->(e:Event)
WITH p, count(e) AS event_count
RETURN p.person_id AS person_id, event_count
ORDER BY event_count DESC
LIMIT 1;

Question: "List all people who appeared in the same event as P456."
MATCH (p:Person {{person_id:'P456'}})-[:APPEARED_IN]->(e:Event)<-[:APPEARED_IN]-(other:Person)
RETURN DISTINCT other.person_id AS co_participant;

Question: "Find the number of events for each event type."
MATCH (e:Event)
WITH e.event_type AS event_type, count(*) AS event_count
RETURN event_type, event_count
ORDER BY event_count DESC;

Question: "Show all events after 2023-01-01."
MATCH (e:Event)
WHERE e.timestamp > datetime('2023-01-01T00:00:00')
RETURN e.event_id, toString(e.timestamp) AS timestamp, e.event_type;

The question is:
{query}
"""


ANSWER_GENERATION_TEMPLATE = """
Task: Summarize the results of a Cypher query into a clear and complete answer.

Database result:
{db_result}

Instructions:
- Do not skip or omit any rows from db_result.
- Include every person_id, event_id, and timestamp explicitly in the output.
- Group events by person_id where possible.
- Output should be structured and complete, not shortened.
- If no results are found, return "No results found."

Question:
{query}

Answer:
"""


QA_TEMPLATE = """Answer the following question based only on the Cypher query results.

Question: {question}
Cypher Results: {context}
Answer:"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "query"], template=CYPHER_GENERATION_TEMPLATE
)


QA_PROMPT = PromptTemplate(input_variables=["question", "context"], template=QA_TEMPLATE)

# ---------- CHAIN ----------
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=QA_PROMPT,
    ans_prompt=ANSWER_GENERATION_TEMPLATE,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
    validate_cypher=True,
)


# ---------- UTILITY ----------
def repair_cypher(query: str) -> str:
    """Clean and sanitize Cypher query to avoid runtime errors."""
    if not query:
        return ""

    # Remove markdown fences
    query = re.sub(r"```cypher\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"```\s*", "", query, flags=re.IGNORECASE)

    # Replace param placeholders with safe hardcoded values
    query = re.sub(r"\$person_id", "'P123'", query, flags=re.IGNORECASE)
    query = re.sub(r"\$vehicle_id", "'V123'", query, flags=re.IGNORECASE)
    query = re.sub(r"\$[a-zA-Z_][a-zA-Z0-9_]*", "'UNKNOWN'", query)

    def fix_union(match):
        parts = match.group(0).split("UNION")
        fixed_parts = []
        for part in parts:
            part = part.strip().rstrip(";")
            # Add a RETURN if missing
            if not re.search(r"\bRETURN\b", part, flags=re.IGNORECASE):
                # Attempt to infer a simple return
                # This returns event_id, timestamp, event_type if an Event node exists
                if "Event" in part:
                    part += " RETURN e.event_id AS event_id, toString(e.timestamp) AS timestamp, e.event_type AS event_type"
                else:
                    part += " RETURN *"
            fixed_parts.append(part)
        return " UNION ".join(fixed_parts)

    query = re.sub(r"(.+UNION.+)", fix_union, query, flags=re.IGNORECASE | re.DOTALL)

    # Fix invalid "OR" inside MATCH clauses
    def fix_or_in_match(match):
        patterns = match.group(1).split(" OR ")
        return "\n".join(f"MATCH {p.strip()}" for p in patterns)

    query = re.sub(
        r"MATCH\s+(.+?)\s+WHERE",
        lambda m: fix_or_in_match(m) + " WHERE",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    query = re.sub(
        r"MATCH\s+(.+)$",
        lambda m: fix_or_in_match(m),
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    query = re.sub(r"\)-\[:([A-Za-z0-9_]+)\]<-\((\w+)\)", r"(<-\[:\1]-\(\2\))", query)
    # Fix invalid ternary operators in ORDER BY
    # Example: e.frame_number IS NOT NULL ? e.frame_number : e.timestamp
    query = re.sub(
        r"(\w+)\.frame_number IS NOT NULL \? \1\.frame_number : \1\.timestamp",
        r"coalesce(\1.frame_number, \1.timestamp)",
        query,
        flags=re.IGNORECASE,
    )

    # Remove trailing semicolons (optional in Neo4j but can cause issues)
    query = query.rstrip(";")
    query = re.sub(r"\s+", " ", query).strip()

    return query.strip()


def make_case_insensitive(query: str) -> str:
    """
    Transform all equality comparisons on string properties to use toLower().
    Example:
    WHERE p.shirt_color = 'Red'  -->  WHERE toLower(p.shirt_color) = toLower('Red')
    """
    # This regex finds patterns like "property = 'value'" or "property='value'"
    pattern = r"(\b\w+\.\w+\b)\s*=\s*'([^']*)'"

    def repl(match):
        prop, val = match.groups()
        # Wrap both sides in toLower()
        return f"toLower({prop}) = toLower('{val}')"

    return re.sub(pattern, repl, query)


# ---------- EXECUTION ----------
def execute_question(user_query: str):
    try:
        schema = graph.get_schema  # property, not callable in your version
        # print("Schema being injected:\n", schema)
        # print("Prompt preview:\n", CYPHER_GENERATION_PROMPT.format(schema=schema, query="dummy"))

        # --- wrap chain call separately ---
        try:
            response = cypher_chain.invoke({"schema": schema, "query": user_query})
            print("DEBUG response:", response)
        except Exception as chain_error:
            print("CHAIN ERROR:", chain_error)
            traceback.print_exc()
            return {"user_query": user_query, "error": str(chain_error)}

        # --- extract intermediate steps safely ---
        steps = response.get("intermediate_steps", []) if isinstance(response, dict) else []
        raw_query = ""

        for step in steps:
            if isinstance(step, dict):
                if "query" in step:
                    raw_query = step["query"]
                    break
                elif "cypher" in step:  # sometimes LLM labels differently
                    raw_query = step["cypher"]
                    break
            elif isinstance(step, str) and step.strip():
                raw_query = step
                break

        print("Raw query:\n", raw_query)
        safe_query = repair_cypher(raw_query)
        print("Safe query:\n", safe_query)

        # Make string comparisons case-insensitive
        safe_query = make_case_insensitive(safe_query)
        print("make_case_insensitive:\n", safe_query)

        # --- run DB query only if valid ---
        db_result = None
        if safe_query and safe_query.upper().startswith(
            ("MATCH", "RETURN", "WITH", "CALL", "SHOW", "CREATE", "MERGE")
        ):
            try:
                db_result = graph.query(safe_query)
                print("DB result:", db_result)
            except Exception as db_error:
                print(f"DB query error: {db_error}")
                traceback.print_exc()
                db_result = {"error": str(db_error)}
        else:
            db_result = {"error": "Invalid query generated"}

        return {
            "user_query": user_query,
            "raw_query": raw_query,
            "sanitized_query": safe_query,
            "chain_answer": response.get("result", "No result")
            if isinstance(response, dict)
            else "No result",
            "db_result": db_result,
            "intermediate_steps": steps,
        }

    except Exception as e:
        print("EXECUTION ERROR:", e)
        traceback.print_exc()
        return {"user_query": user_query, "error": str(e)}


# ---------- FASTAPI ----------
app = FastAPI(title="Agentic Video Intelligence API", version="1.0.0")


class Question(BaseModel):
    query: str


@app.post("/ask")
async def ask_question(question: Question):
    return execute_question(question.query)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
