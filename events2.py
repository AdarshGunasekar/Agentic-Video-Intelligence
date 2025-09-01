import json
import random
import datetime

persons = ["P123", "P456", "P789"]
vehicles = ["V101", "V102", "V103"]
cameras = ["Cam1", "Cam2", "Cam3", "Cam4"]
locations = ["Lobby", "Parking", "Entrance", "Hallway"]
event_types = ["person_crossing", "vehicle_entry", "vehicle_exit", "person_appearance"]

def generate_mock_events(num_events=20):
    events = []
    for i in range(num_events):
        timestamp = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 1000))
        frame_number = random.randint(1000, 50000)
        video_time = f"0:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
        event_type = random.choice(event_types)
        camera_id = random.choice(cameras)
        location = random.choice(locations)
        video_file = f"{location.lower()}_{camera_id}.mp4"
        confidence = round(random.uniform(0.7, 0.99), 2)

        if "person" in event_type:
            entity = "Person"
            person_id = random.choice(persons)
            additional_info = {
                "confidence": confidence,
                "shirt_color": random.choice(["red", "blue", "green"])
            }
            vehicle_id = None
        else:
            entity = "Vehicle"
            vehicle_id = random.choice(vehicles)
            additional_info = {
                "confidence": confidence,
                "license_plate": f"KA-{random.randint(10,99)}-{random.randint(1000,9999)}"
            }
            person_id = None

        events.append({
            "event_id": 100 + i,
            "entity": entity,
            "person_id": person_id,
            "vehicle_id": vehicle_id,
            "camera_id": camera_id,
            "location": location,
            "timestamp": timestamp.isoformat(),
            "video_file": video_file,
            "frame_number": frame_number,
            "video_time": video_time,
            "event_type": event_type,
            "additional_info": additional_info
        })

    return events

if __name__ == "__main__":
    mock_events = generate_mock_events(30)
    with open("graph2_events.json", "w") as f:
        json.dump(mock_events, f, indent=4)
    print("Mock events saved to mock_events.json")
