import json
import random
from datetime import datetime, timedelta

# -------------------------------
# Schema IDs and attribute pools
# -------------------------------
person_ids = [f"P{100+i}" for i in range(10)]  # P100-P109
vehicle_ids = [f"V20{i}" for i in range(6)]  # V200-V205
camera_ids = ["Cam1","Cam2","Cam3","Cam4","Cam5"]
locations = ["Lobby","Parking Lot","Entrance","Hallway","Staircase"]
shirt_colors = ["red","blue","green","yellow","white","black"]
event_types = ["person_appearance","person_crossing","vehicle_entry","vehicle_exit"]
video_files = [
    "entrance_Cam2.mp4",
    "lobby_Cam1.mp4",
    "entrance_Cam1.mp4",
    "lobby_Cam4.mp4",
    "lobby_Cam2.mp4",
    "hallway_Cam4.mp4",
    "entrance_Cam5.mp4",
    "hallway_Cam1.mp4",
    "parking_Cam2.mp4",
    "parking_Cam5.mp4",
    "staircase_Cam3.mp4",
    "staircase_Cam5.mp4",
    "parking_Cam4.mp4",
    "entrance_Cam4.mp4",
    "staircase_Cam2.mp4",
    "lobby_Cam3.mp4",
    "hallway_Cam3.mp4",
    "parking_Cam1.mp4",
    "entrance_Cam3.mp4"
]



# Event timestamps range
start_date = datetime(2025,9,4)
end_date = datetime(2025,9,11)
delta = end_date - start_date

# -------------------------------
# Helper functions
# -------------------------------

def random_timestamp():
    rand_days = random.randint(0, delta.days)
    rand_seconds = random.randint(0, 86399)
    dt = start_date + timedelta(days=rand_days, seconds=rand_seconds)
    return dt.isoformat() + "Z"

def choose_two_unique(pool):
    a, b = random.sample(pool, 2)
    return a, b

def generate_nl_paraphrases(template, substitutions, n=3):
    """
    Generate n simple paraphrases from a template.
    """
    base = template.format(**substitutions)
    paraphrases = [base]
    for _ in range(n-1):
        p = base
        # simple variations: reorder, change verbs
        p = p.replace("List all", random.choice(["Show all","Find all","Which"]))
        p = p.replace("appeared", random.choice(["were seen","co-appeared","occurred"]))
        paraphrases.append(p)
    return paraphrases

# -------------------------------
# Logical query types
# -------------------------------

query_templates = [
    # single-entity
    {
        "difficulty":"easy",
        "cypher_template":"MATCH (p:Person {{person_id:'{person_id}'}})-[:APPEARED_IN]->(e:Event) RETURN p.person_id, e.event_id",
        "nl_template":"List all events where Person {person_id} appeared."
    },
    {
        "difficulty":"easy",
        "cypher_template":"MATCH (v:Vehicle {{vehicle_id:'{vehicle_id}'}})-[:APPEARED_IN]->(e:Event) RETURN v.vehicle_id, e.event_id",
        "nl_template":"List all events where Vehicle {vehicle_id} appeared."
    },
    # co-appearance
    {
        "difficulty":"complex",
        "cypher_template":"MATCH (p1:Person {{person_id:'{person_id_1}'}})-[:APPEARED_IN]->(e:Event)<-[:APPEARED_IN]-(p2:Person {{person_id:'{person_id_2}'}}) RETURN e.event_id, p1.person_id, p2.person_id",
        "nl_template":"List all events where Person {person_id_1} appeared with Person {person_id_2}."
    },
    # entity + attribute
    {
        "difficulty":"medium",
        "cypher_template":"MATCH (p:Person {{person_id:'{person_id}'}})-[:APPEARED_IN]->(e:Event) WHERE e.shirt_color='{shirt_color}' RETURN p.person_id, e.event_id, e.shirt_color",
        "nl_template":"Find events where Person {person_id} wore a {shirt_color} shirt."
    },
    # entity + vehicle + camera + confidence
    {
        "difficulty":"complex",
        "cypher_template":"MATCH (p:Person {{person_id:'{person_id}'}})-[:APPEARED_IN]->(e:Event)<-[:APPEARED_IN]-(v:Vehicle {{vehicle_id:'{vehicle_id}'}})-[:CAPTURED_BY]->(c:Camera {{camera_id:'{camera_id}'}}) WHERE e.confidence > {conf} RETURN p.person_id, v.vehicle_id, e.event_id, e.confidence, c.camera_id",
        "nl_template":"List all events where Person {person_id} co-appeared with Vehicle {vehicle_id} captured by Camera {camera_id} with confidence above {conf}."
    },
    # time-based query
    {
        "difficulty":"medium",
        "cypher_template":"MATCH (v:Vehicle {{vehicle_id:'{vehicle_id}'}})-[:APPEARED_IN]->(e:Event) WHERE e.timestamp >= '{start_time}' AND e.timestamp <= '{end_time}' RETURN v.vehicle_id, e.event_id, e.timestamp",
        "nl_template":"List all events where Vehicle {vehicle_id} appeared between {start_time} and {end_time}."
    }
]

# -------------------------------
# Generate dataset
# -------------------------------
num_samples = 1200
dataset = []

for _ in range(num_samples):
    tmpl = random.choice(query_templates)
    
    # Prepare substitutions
    substitutions = {}
    if "{person_id}" in tmpl["cypher_template"]:
        substitutions["person_id"] = random.choice(person_ids)
    if "{person_id_1}" in tmpl["cypher_template"]:
        substitutions["person_id_1"], substitutions["person_id_2"] = choose_two_unique(person_ids)
    if "{person_id_2}" in tmpl["cypher_template"]:
        substitutions["person_id_2"] = substitutions.get("person_id_2", random.choice(person_ids))
    if "{vehicle_id}" in tmpl["cypher_template"]:
        substitutions["vehicle_id"] = random.choice(vehicle_ids)
    if "{camera_id}" in tmpl["cypher_template"]:
        substitutions["camera_id"] = random.choice(camera_ids)
    if "{shirt_color}" in tmpl["cypher_template"]:
        substitutions["shirt_color"] = random.choice(shirt_colors)
    if "{conf}" in tmpl["cypher_template"]:
        substitutions["conf"] = round(random.uniform(0.5,1.0),2)
    if "{start_time}" in tmpl["cypher_template"]:
        substitutions["start_time"] = random_timestamp()
    if "{end_time}" in tmpl["cypher_template"]:
        substitutions["end_time"] = random_timestamp()
    if "{video_file}" in tmpl["cypher_template"] or "{video_file}" in tmpl["nl_template"]:
        substitutions["video_file"] = random.choice(video_files)

    # Generate NL paraphrases
    nl_paraphrases = generate_nl_paraphrases(tmpl["nl_template"], substitutions, n=3)
    
    # Generate Cypher
    cypher_query = tmpl["cypher_template"].format(**substitutions)
    
    # Add multiple paraphrases to dataset
    for nl in nl_paraphrases:
        dataset.append({
            "difficulty": tmpl["difficulty"],
            "question": nl,
            "cypher": cypher_query
        })

# -------------------------------
# Save to JSON
# -------------------------------
with open("nl_cypher_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} NL → Cypher pairs in nl_cypher_dataset.json")
