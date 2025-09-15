import json
import random
import datetime

# Extended lists
persons = [f"P{100+i}" for i in range(10)]  # P100..P109
vehicles = [f"V{200+i}" for i in range(6)]  # V200..V205
cameras = ["Cam1", "Cam2", "Cam3", "Cam4", "Cam5"]
locations = ["Lobby", "Parking", "Entrance", "Hallway", "Staircase"]
colors = ["red", "blue", "green", "yellow", "black", "white", "grey", "orange"]

# Mapping entity -> allowed event types
allowed_event_types = {
    "Person": ["person_crossing", "person_appearance"],
    "Vehicle": ["vehicle_entry", "vehicle_exit"],
}

def generate_event(event_id, entity_type, person_id=None, vehicle_id=None):
    """Helper to generate a single event with correct event type + description."""
    timestamp = datetime.datetime.now() - datetime.timedelta(
        minutes=random.randint(0, 60 * 24 * 7)
    )
    frame_number = random.randint(1000, 50000)
    video_time = f"0:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
    event_type = random.choice(allowed_event_types[entity_type])  # ✅ Corrected
    camera_id = random.choice(cameras)
    location = random.choice(locations)
    video_file = f"{location.lower()}_{camera_id}.mp4"
    confidence = round(random.uniform(0.7, 0.99), 2)

    if entity_type == "Person":
        if not person_id:
            person_id = random.choice(persons)
        shirt_color = random.choice(colors).lower()
        license_plate = None
        vehicle_id = None
        vehicle_color = None
        additional_info = {"confidence": confidence, "shirt_color": shirt_color}

        description = (
            f"At {timestamp}, a person (ID {person_id}) wearing a {shirt_color} shirt "
            f"was observed in the {location} by {camera_id} during {event_type}. "
            f"Confidence: {confidence}."
        )

    else:  # Vehicle
        if not vehicle_id:
            vehicle_id = random.choice(vehicles)
        license_plate = f"KA-{random.randint(10,99)}-{random.randint(1000,9999)}"
        vehicle_color = random.choice(colors).lower()
        shirt_color = None
        person_id = None
        additional_info = {
            "confidence": confidence,
            "license_plate": license_plate,
            "vehicle_color": vehicle_color,
        }

        description = (
            f"At {timestamp}, a {vehicle_color} vehicle (ID {vehicle_id}, "
            f"license plate {license_plate}) was captured in the {location} by {camera_id} "
            f"during {event_type}. Confidence: {confidence}."
        )

    return {
        "event_id": event_id,
        "entity": entity_type,
        "person_id": person_id,
        "vehicle_id": vehicle_id,
        "camera_id": camera_id,
        "location": location,
        "timestamp": timestamp.isoformat(),
        "video_file": video_file,
        "frame_number": frame_number,
        "video_time": video_time,
        "event_type": event_type,
        "confidence": confidence,
        "shirt_color": shirt_color if entity_type == "Person" else None,
        "license_plate": license_plate if entity_type == "Vehicle" else None,
        "vehicle_color": vehicle_color if entity_type == "Vehicle" else None,
        "additional_info": additional_info,
        "description": description,  # ✅ Added natural language description
    }

def generate_mock_events(num_events=200):
    events = []
    event_id = 100

    # Ensure every person appears at least once
    for p in persons:
        events.append(generate_event(event_id, "Person", person_id=p))
        event_id += 1

    # Ensure every vehicle appears at least once
    for v in vehicles:
        events.append(generate_event(event_id, "Vehicle", vehicle_id=v))
        event_id += 1

    # Fill the rest randomly
    while len(events) < num_events:
        entity_type = random.choice(["Person", "Vehicle"])
        events.append(generate_event(event_id, entity_type))
        event_id += 1

    return events

if __name__ == "__main__":
    mock_events = generate_mock_events(200)
    with open("graph2_events.json", "w") as f:
        json.dump(mock_events, f, indent=4)
    print("✅ 200 mock events with descriptions saved to graph2_events.json")
