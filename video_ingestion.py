import cv2
import torch
import json
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from paddleocr import PaddleOCR
from datetime import datetime

# ------------------ Load Models ------------------

# YOLO (includes built-in ByteTrack)
yolo = YOLO("yolov8n.pt")  # pretrained COCO: person, car, truck etc.

# Face recognition
face_app = FaceAnalysis(name="buffalo_l")  
face_app.prepare(ctx_id=0, det_size=(640, 640))

# OCR for license plates
ocr = PaddleOCR(use_angle_cls=True, lang='en')


# ------------------ Processing Function ------------------
def process_video(video_path, camera_id="Cam1", location="Gate_1"):
    cap = cv2.VideoCapture(video_path)
    event_id = 1
    events = []

    # Run YOLO in tracking mode with ByteTrack
    results = yolo.track(source=video_path, stream=True, tracker="bytetrack.yaml")

    for r in results:
        frame = r.orig_img
        timestamp = datetime.now().isoformat()

        # --- Loop over YOLO detections ---
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Person → Face Recognition
            if label == "person":
                faces = face_app.get(frame)
                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    name = "Unknown"  # TODO: compare face.embedding to DB
                    events.append({
                        "event_id": event_id,
                        "entity": "Person",
                        "name": name,
                        "camera_id": camera_id,
                        "location": location,
                        "timestamp": timestamp,
                        "additional_info": {
                            "confidence": float(face.det_score),
                            "bbox": [fx1, fy1, fx2, fy2]
                        }
                    })
                    event_id += 1

            # Vehicle → License Plate OCR
            if label in ["car", "truck"]:
                plate_crop = frame[y1:y2, x1:x2]
                ocr_result = ocr.ocr(plate_crop, cls=True)
                plate = None
                if ocr_result and len(ocr_result[0]) > 0:
                    plate = ocr_result[0][0][1][0]  # best OCR guess

                events.append({
                    "event_id": event_id,
                    "entity": "Vehicle",
                    "name": label,
                    "camera_id": camera_id,
                    "location": location,
                    "timestamp": timestamp,
                    "additional_info": {
                        "license_plate": plate,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    }
                })
                event_id += 1

    cap.release()

    # Save JSON
    with open("events.json", "w") as f:
        json.dump(events, f, indent=2)

    print(f"✅ Generated {len(events)} events and saved to events.json")


# ------------------ Example Run ------------------
if __name__ == "__main__":
    process_video("sample_video.mp4")

