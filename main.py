"""
FastAPI backend for Face-Recognition Attendance Tracker.
Uses OpenCV for face detection and DeepFace for face embeddings/verification.
No images are stored — only face embeddings in SQLite.
"""

import base64
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from deepface import DeepFace
import logging

import database as db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Attendance Tracker")

# OpenCV face detector (Haar cascade – ships with opencv)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# DeepFace model/backend config (Facenet is accurate; opencv backend avoids dlib)
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"
VERIFICATION_THRESHOLD = 0.40  # cosine distance threshold for Facenet


@app.on_event("startup")
def startup():
    db.init_db()
    # Warm up the model on first load
    logger.info("Warming up DeepFace model (first load may take a moment)...")
    try:
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        DeepFace.represent(
            dummy, model_name=MODEL_NAME, detector_backend="skip", enforce_detection=False
        )
        logger.info("DeepFace model ready.")
    except Exception as e:
        logger.warning(f"Warm-up note: {e}")


# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


# ─── Pydantic models ────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    image: str  # base64-encoded JPEG frame


class RecognizeRequest(BaseModel):
    image: str  # base64-encoded JPEG frame


# ─── Helpers ─────────────────────────────────────────────────────────────────
def decode_image(b64: str) -> np.ndarray:
    """Decode a base64-encoded image string to an OpenCV BGR numpy array."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def detect_faces_opencv(img: np.ndarray):
    """Detect faces using OpenCV Haar cascade. Returns list of (x,y,w,h) rects."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return faces


def get_embedding(face_img: np.ndarray) -> list:
    """Extract face embedding using DeepFace with detection skipped (already cropped)."""
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        rgb,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
    )
    if isinstance(result, list) and len(result) > 0:
        return result[0]["embedding"]
    raise ValueError("Could not extract embedding")


def cosine_distance(a: list, b: list) -> float:
    """Compute cosine distance between two embedding vectors."""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/register")
def register_face(req: RegisterRequest):
    """Register a new person: detect face, extract embedding, save to DB."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    try:
        img = decode_image(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    faces = detect_faces_opencv(img)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image. Please try again.")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Please ensure only one face is visible.")

    x, y, w, h = faces[0]
    # Add padding around the face for better embedding
    pad = int(0.2 * max(w, h))
    y1 = max(0, y - pad)
    y2 = min(img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img.shape[1], x + w + pad)
    face_crop = img[y1:y2, x1:x2]

    try:
        embedding = get_embedding(face_crop)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not extract face features: {e}")

    user_id = db.add_user(req.name.strip(), embedding)
    return {"success": True, "user_id": user_id, "message": f"{req.name.strip()} registered successfully!"}


@app.post("/api/recognize")
def recognize_faces(req: RecognizeRequest):
    """Recognize faces in a frame and record attendance for matches."""
    try:
        img = decode_image(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    faces = detect_faces_opencv(img)

    if len(faces) == 0:
        return {"recognized": [], "unknown_count": 0, "face_locations": []}

    # Load all known users
    users = db.get_all_users()
    if not users:
        face_locs = [[int(y), int(x + w), int(y + h), int(x)] for (x, y, w, h) in faces]
        return {"recognized": [], "unknown_count": len(faces), "face_locations": face_locs}

    known_embeddings = [u["encoding"] for u in users]
    known_names = [u["name"] for u in users]
    known_ids = [u["id"] for u in users]

    recognized = []
    unknown_count = 0

    for (x, y, w, h) in faces:
        # Crop with padding
        pad = int(0.2 * max(w, h))
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        face_crop = img[y1:y2, x1:x2]

        try:
            embedding = get_embedding(face_crop)
        except Exception:
            unknown_count += 1
            continue

        # Compare against all known faces
        distances = [cosine_distance(embedding, ke) for ke in known_embeddings]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]

        if min_dist < VERIFICATION_THRESHOLD:
            name = known_names[min_idx]
            user_id = known_ids[min_idx]
            att_id = db.record_attendance(user_id, name)
            # face location in [top, right, bottom, left] format for frontend
            recognized.append({
                "name": name,
                "user_id": user_id,
                "confidence": round(1.0 - min_dist, 2),
                "location": [int(y), int(x + w), int(y + h), int(x)],
                "already_recorded": att_id == -1,
            })
        else:
            unknown_count += 1

    face_locs = [[int(y), int(x + w), int(y + h), int(x)] for (x, y, w, h) in faces]

    return {
        "recognized": recognized,
        "unknown_count": unknown_count,
        "face_locations": face_locs,
    }


@app.get("/api/attendance")
def get_attendance(date: str | None = Query(None)):
    """Retrieve attendance records, optionally filtered by date (YYYY-MM-DD)."""
    records = db.get_attendance(date)
    return {"records": records}


@app.get("/api/users")
def list_users():
    """List all registered users (without encoding data)."""
    users = db.get_users_list()
    return {"users": users}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    """Delete a registered user."""
    deleted = db.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "message": "User deleted successfully"}


@app.get("/api/stats")
def get_stats():
    """Return dashboard statistics."""
    return db.get_stats()
