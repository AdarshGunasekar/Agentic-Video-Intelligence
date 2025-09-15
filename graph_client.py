# graph_client.py
import datetime
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import json

load_dotenv()


class GraphClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "adhu@2580")
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    # ---- Constraints for fast MERGE ----
    def ensure_constraints(self):
        cyphers = [
            # Unique constraints (already exist)
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.vehicle_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Camera) REQUIRE c.camera_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",

            # Additional indexes for faster filtering and range queries
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.vehicle_color)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.shirt_color)",
        ]
        with self.driver.session() as sess:
            for c in cyphers:
                sess.run(c)
        print("✅ Constraints and indexes ensured for fast MERGE & queries")
    # ---- Wipe the whole graph ----
    def wipe(self):
        with self.driver.session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")
        print("⚠️ Wiped entire Neo4j database (all nodes + relationships).")

    # ---- Ingest events ----
    def ingest_events(self, events: List[Dict[str, Any]]):
        with self.driver.session() as sess:
            last_event_by_entity: Dict[str, str] = {}

            for ev in events:
                event_id = ev["event_id"]
                entity = ev.get("entity")  # "Person" or "Vehicle"
                timestamp = ev.get("timestamp")
                camera_id = ev.get("camera_id")
                location = ev.get("location")
                event_type = ev.get("event_type")
                video_file = ev.get("video_file")
                frame_number = ev.get("frame_number")
                video_time = ev.get("video_time")

                # flatten additional_info
                ai = ev.get("additional_info", {}) or {}
                confidence = ai.get("confidence")
                shirt_color = ai.get("shirt_color")
                license_plate = ai.get("license_plate")
                vehicle_color = ai.get("vehicle_color")
                description = ev.get("description")
                # Upsert Event
                sess.run(
                """
                MERGE (e:Event {event_id: $event_id})
                SET e.timestamp = datetime($timestamp),
                    e.event_type = $event_type,
                    e.video_file = $video_file,
                    e.frame_number = $frame_number,
                    e.video_time = $video_time,
                    e.confidence = $confidence,
                    e.shirt_color = $shirt_color,
                    e.license_plate = $license_plate,
                    e.vehicle_color = $vehicle_color,
                    e.description = $description
                """,
                {
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "video_file": video_file,
                    "frame_number": frame_number,
                    "video_time": video_time,
                    "confidence": confidence,
                    "shirt_color": shirt_color.lower() if shirt_color else None,
                    "license_plate": license_plate,
                    "vehicle_color": vehicle_color.lower() if vehicle_color else None,
                    "description": description,
                },
                )


                # Camera
                if camera_id:
                    sess.run(
                        """
                        MERGE (c:Camera {camera_id: $camera_id})
                        MERGE (e:Event {event_id: $event_id})
                        MERGE (e)-[:CAPTURED_BY {confidence: $confidence}]->(c)
                    """,
                        {"camera_id": camera_id, "event_id": event_id, "confidence": confidence},
                    )

                # Location
                if location:
                    sess.run(
                        """
                        MERGE (l:Location {name: $name})
                        MERGE (e:Event {event_id: $event_id})
                        MERGE (e)-[:AT]->(l)
                    """,
                        {"name": location, "event_id": event_id},
                    )

                # Entity (Person or Vehicle)
                if entity == "Person":
                    pid = ev.get("person_id", "UNKNOWN")
                    sess.run(
                        """
                        MERGE (p:Person {person_id: $pid})
                        MERGE (e:Event {event_id: $event_id})
                        MERGE (p)-[:APPEARED_IN]->(e)
                    """,
                        {"pid": pid, "event_id": event_id},
                    )
                    key = f"P::{pid}"

                else:  # Vehicle
                    vid = ev.get("vehicle_id", "UNKNOWN")
                    vtype = ev.get("type")
                    sess.run(
                        """
                        MERGE (v:Vehicle {vehicle_id: $vid})
                        SET v.type = $vtype,
                            v.license_plate = $license_plate
                        MERGE (e:Event {event_id: $event_id})
                        MERGE (v)-[:APPEARED_IN]->(e)
                    """,
                        {
                            "vid": vid,
                            "vtype": vtype,
                            "license_plate": license_plate,
                            "event_id": event_id,
                        },
                    )
                    key = f"V::{vid}"

                # Temporal chain
                if timestamp:
                    prev = last_event_by_entity.get(key)
                    if prev:
                        sess.run(
                            """
                            MATCH (prev:Event {event_id: $prev_id})
                            MATCH (curr:Event {event_id: $curr_id})
                            MERGE (prev)-[:FOLLOWS]->(curr)
                        """,
                            {"prev_id": prev, "curr_id": event_id},
                        )
                    last_event_by_entity[key] = event_id

    # ---- Load and ingest directly from file ----
    def ingest_from_file(self, file_path: str, reset: bool = False):
        with open(file_path, "r") as f:
            events = json.load(f)
        if reset:
            self.wipe()
        self.ensure_constraints()
        self.ingest_events(events)
        print(f"✅ Ingested {len(events)} events from {file_path}")

    # ---- Helpers for common queries ----
    def person_trail(self, person_id: str, start: Optional[str] = None, end: Optional[str] = None):
        q = """
        MATCH (p:Person {person_id: $pid})-[:APPEARED_IN]->(e:Event)-[:AT]->(l:Location)
        OPTIONAL MATCH (e)-[:CAPTURED_BY]->(c:Camera)
        WHERE ($start IS NULL OR e.timestamp >= datetime($start))
          AND ($end   IS NULL OR e.timestamp <= datetime($end))
        RETURN e.event_id AS event_id, toString(e.timestamp) AS ts, l.name AS location,
               c.camera_id AS camera_id, e.video_file AS video_file,
               e.frame_number AS frame_number, e.video_time AS video_time,
               e.shirt_color AS shirt_color
        ORDER BY e.timestamp
        """
        with self.driver.session() as sess:
            return [dict(r) for r in sess.run(q, {"pid": person_id, "start": start, "end": end})]

    def vehicle_trail(
        self, vehicle_id: str, start: Optional[str] = None, end: Optional[str] = None
    ):
        q = """
        MATCH (v:Vehicle {vehicle_id: $vid})-[:APPEARED_IN]->(e:Event)-[:AT]->(l:Location)
        OPTIONAL MATCH (e)-[:CAPTURED_BY]->(c:Camera)
        WHERE ($start IS NULL OR e.timestamp >= datetime($start))
          AND ($end   IS NULL OR e.timestamp <= datetime($end))
        RETURN e.event_id AS event_id, toString(e.timestamp) AS ts, l.name AS location,
               c.camera_id AS camera_id, e.video_file AS video_file,
               e.frame_number AS frame_number, e.video_time AS video_time,
               e.vehicle_color AS vehicle_color
        ORDER BY e.timestamp
        """
        with self.driver.session() as sess:
            return [dict(r) for r in sess.run(q, {"vid": vehicle_id, "start": start, "end": end})]

    # ---- Query red vehicles by color ----
    def red_vehicles_timestamps(self):
        """Get timestamps of all red vehicles"""
        q = """
        MATCH (v:Vehicle)-[:APPEARED_IN]->(e:Event)
        WHERE toLower(e.vehicle_color) = 'red'
        RETURN e.event_id AS event_id, toString(e.timestamp) AS timestamp, 
               v.vehicle_id AS vehicle_id, e.vehicle_color AS color, e.location AS location
        ORDER BY e.timestamp
        """
        with self.driver.session() as sess:
            return [dict(r) for r in sess.run(q)]

    def run_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        with self.driver.session() as sess:
            return [dict(r) for r in sess.run(cypher, params or {})]
