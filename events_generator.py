import json
import random
from datetime import datetime, timedelta

# Attributes to randomize
names = ["John Doe", "Alice Smith", "Robert Brown", "Emily Davis", "Michael Johnson",
         "Sophia Miller", "David Wilson", "Olivia Martinez", "James Taylor", "Emma Thomas"]
entities = ["Person", "Vehicle"]
locations = ["Gate", "Lobby", "Parking Lot", "Hallway", "Exit", "Cafeteria"]
vehicle_types = ["Car", "Truck", "Bike"]
plates = ["ABC123", "XYZ987", "LMN456", "JKL321", "DEF654", "PQR789", "UVW111", "TUV222"]

# Generate 1000 events
events = []
start_time = datetime(2025, 8, 18, 9, 0, 0)

for event_id in range(1, 1001):
    entity = random.choice(entities)
    timestamp = (start_time + timedelta(seconds=event_id * 30)).isoformat()

    if entity == "Person":
        event = {
            "event_id": event_id,
            "entity": entity,
            "name": random.choice(names),
            "camera_id": f"Cam{random.randint(1, 4)}",
            "location": random.choice(locations),
            "timestamp": timestamp,
            "additional_info": {
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "age_estimate": random.randint(18, 60),
                "gender": random.choice(["Male", "Female"])
            }
        }
    else:  # Vehicle
        event = {
            "event_id": event_id,
            "entity": entity,
            "name": random.choice(vehicle_types),
            "camera_id": f"Cam{random.randint(1, 4)}",
            "location": random.choice(locations),
            "timestamp": timestamp,
            "additional_info": {
                "license_plate": random.choice(plates),
                "confidence": round(random.uniform(0.7, 0.99), 2)
            }
        }

    events.append(event)

# Save to file
with open("mock1_events.json", "w") as f:
    json.dump(events, f, indent=2)

len(events)
