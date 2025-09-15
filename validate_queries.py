import json
from pathlib import Path
from langchain_community.graphs import Neo4jGraph

# === Neo4j connection ===
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "adhu@2580"

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# === Files ===
INPUT_FILE = Path("sample.json")        # input queries
OUTPUT_FILE = Path("sample_valid.json") # valid queries (append mode)

def validate_cypher(cypher: str) -> bool:
    """Try running Cypher query in Neo4j. Returns True if valid, False otherwise."""
    try:
        query = cypher.strip().rstrip(";")
        if "LIMIT" not in query.upper():
            query += " LIMIT 1"
        _ = graph.query(query)
        return True
    except Exception as e:
        print(f"❌ Cypher failed: {cypher}\n   Error: {e}\n")
        return False

def main():
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ Expected JSON list of queries")
        return

    print(f"🚀 Validating {len(data)} stored queries...")

    # Load existing valid queries if file exists
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_valid_queries = json.load(f)
            if not isinstance(all_valid_queries, list):
                all_valid_queries = []
        except json.JSONDecodeError:
            all_valid_queries = []
    else:
        all_valid_queries = []

    new_valid = []
    invalid_count = 0

    for i, item in enumerate(data, start=1):
        difficulty = item.get("difficulty", "unknown")
        question = item.get("question", "N/A")
        cypher = item.get("cypher")

        print(f"\n[{i}] 🔎 Difficulty={difficulty} | Question={question}")

        if not cypher:
            print("   ❌ Missing Cypher query.")
            invalid_count += 1
            continue

        if validate_cypher(cypher):
            print("   ✅ Query is valid.")
            new_valid.append(item)
        else:
            print("   ❌ Query invalid.")
            invalid_count += 1

    # Append new valid queries
    all_valid_queries.extend(new_valid)

    # Save updated file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_valid_queries, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"New valid queries: {len(new_valid)}")
    print(f"Invalid queries: {invalid_count}")
    print(f"Total stored in {OUTPUT_FILE}: {len(all_valid_queries)}")

if __name__ == "__main__":
    main()
