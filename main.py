"""
FastAPI backend for Face-Recognition Attendance Tracker.
Uses DeepFace for face detection + embeddings/verification.
Includes blink-based liveness detection and LBP texture anti-spoofing.
No images are stored — only face embeddings in SQLite.
"""

import base64
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from deepface import DeepFace
import logging
import io

import database as db
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Attendance Tracker")

# OpenCV Haar cascades (kept as fallback for blink/eye detection only)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# DeepFace model/backend config
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"  # DeepFace's built-in opencv detector (DNN-based, NOT Haar)
DISTANCE_METRIC = "cosine"
VERIFICATION_THRESHOLD = 0.40  # cosine distance threshold for Facenet

# Liveness thresholds
MIN_BLINKS_REQUIRED = 1    # Minimum blinks detected across the frame batch
EYE_AR_THRESHOLD = 0.22   # Below this → closed eyes
LBP_SPOOF_THRESHOLD = 45  # Texture variance below this → likely a flat surface


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


class MultiRegisterRequest(BaseModel):
    name: str
    images: list[str]  # list of 3 base64-encoded JPEG frames


class RecognizeRequest(BaseModel):
    image: str  # base64-encoded JPEG frame


class LivenessCheckRequest(BaseModel):
    frames: list[str]  # 5-8 base64 frames captured in quick sequence


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


def extract_embedding_deepface(img: np.ndarray) -> dict | None:
    """Use DeepFace to detect face AND extract embedding in one shot.
    Returns dict with 'embedding' and 'facial_area' keys, or None if no face found.
    This is far more robust than manual Haar cascade detection."""
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = DeepFace.represent(
            rgb,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
        if isinstance(results, list) and len(results) > 0:
            return results[0]  # {"embedding": [...], "facial_area": {"x":..., "y":..., "w":..., "h":...}}
        return None
    except Exception as e:
        logger.debug(f"DeepFace detection failed: {e}")
        return None


def extract_embedding_fallback(img: np.ndarray) -> dict | None:
    """Fallback: skip face detection entirely and extract embedding from the whole frame.
    Useful when the face is obvious (e.g., registration with webcam centered on face)."""
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = DeepFace.represent(
            rgb,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        if isinstance(results, list) and len(results) > 0:
            h, w = img.shape[:2]
            result = results[0]
            result["facial_area"] = {"x": 0, "y": 0, "w": w, "h": h}
            return result
        return None
    except Exception as e:
        logger.debug(f"Fallback embedding extraction failed: {e}")
        return None


def smart_extract_embedding(img: np.ndarray) -> dict | None:
    """Try DeepFace detection first, fall back to skip-detection mode."""
    result = extract_embedding_deepface(img)
    if result is not None:
        return result
    logger.info("Primary face detection failed, trying fallback (skip detection)...")
    return extract_embedding_fallback(img)


def detect_faces_opencv(img: np.ndarray):
    """Detect faces using Haar cascade. Only used for blink/liveness detection.
    Uses CLAHE for better detection in poor lighting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20)
        )
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


def check_duplicate_face(new_embedding: list) -> dict | None:
    """Compare a new embedding against all existing users.
    Returns the user dict if a match is found, else None."""
    users = db.get_all_users()
    for user in users:
        dist = cosine_distance(new_embedding, user["encoding"])
        if dist < VERIFICATION_THRESHOLD:
            return user
    return None


def crop_face_with_padding(img: np.ndarray, x, y, w, h, pad_ratio=0.2):
    """Crop a face region with padding."""
    pad = int(pad_ratio * max(w, h))
    y1 = max(0, y - pad)
    y2 = min(img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img.shape[1], x + w + pad)
    return img[y1:y2, x1:x2]


# ─── Liveness Detection Helpers ──────────────────────────────────────────────

def compute_eye_aspect_ratio(eye_region_gray: np.ndarray) -> float:
    """Approximate EAR using the eye region bounding box aspect ratio.
    Real EAR uses 6 landmarks; this approximation uses height/width ratio."""
    h, w = eye_region_gray.shape[:2]
    if w == 0:
        return 0.0
    return h / w


def detect_blinks_in_frames(frames: list[np.ndarray]) -> int:
    """Count blinks across a sequence of frames using eye detection.
    A blink = eyes detected in frame N, NOT detected in frame N+1, detected in frame N+2."""
    eyes_open_sequence = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        if len(faces) == 0:
            eyes_open_sequence.append(None)  # No face
            continue

        x, y, w, h = faces[0]
        # Only search top half of face for eyes
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
        eyes_open_sequence.append(len(eyes) >= 2)  # True if 2+ eyes found

    # Count transitions: open→closed→open = 1 blink
    blinks = 0
    i = 0
    while i < len(eyes_open_sequence) - 2:
        if eyes_open_sequence[i] is None:
            i += 1
            continue
        if eyes_open_sequence[i] == True:
            # Look for a closed frame
            j = i + 1
            while j < len(eyes_open_sequence) and eyes_open_sequence[j] is None:
                j += 1
            if j < len(eyes_open_sequence) and eyes_open_sequence[j] == False:
                # Look for re-open
                k = j + 1
                while k < len(eyes_open_sequence) and eyes_open_sequence[k] is None:
                    k += 1
                if k < len(eyes_open_sequence) and eyes_open_sequence[k] == True:
                    blinks += 1
                    i = k + 1
                    continue
        i += 1

    return blinks


def compute_lbp_variance(face_gray: np.ndarray) -> float:
    """Compute LBP (Local Binary Pattern) texture variance.
    Real faces have high texture variance; flat photos have low variance."""
    if face_gray.shape[0] < 10 or face_gray.shape[1] < 10:
        return 0.0

    face_resized = cv2.resize(face_gray, (128, 128))
    lbp = np.zeros_like(face_resized, dtype=np.uint8)
    rows, cols = face_resized.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = face_resized[i, j]
            binary = 0
            binary |= (1 << 7) if face_resized[i-1, j-1] >= center else 0
            binary |= (1 << 6) if face_resized[i-1, j]   >= center else 0
            binary |= (1 << 5) if face_resized[i-1, j+1] >= center else 0
            binary |= (1 << 4) if face_resized[i,   j+1] >= center else 0
            binary |= (1 << 3) if face_resized[i+1, j+1] >= center else 0
            binary |= (1 << 2) if face_resized[i+1, j]   >= center else 0
            binary |= (1 << 1) if face_resized[i+1, j-1] >= center else 0
            binary |= (1 << 0) if face_resized[i,   j-1] >= center else 0
            lbp[i, j] = binary

    return float(np.var(lbp))


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/register")
def register_face(req: RegisterRequest):
    """Register a new person: detect face, extract embedding, save to DB.
    Uses DeepFace for robust face detection + embedding in one step."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    try:
        img = decode_image(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    result = smart_extract_embedding(img)
    if result is None:
        raise HTTPException(status_code=400, detail="No face detected in the image. Please ensure your face is clearly visible, well-lit, and centered in the camera.")

    embedding = result["embedding"]
    
    # Check for duplicate face
    duplicate = check_duplicate_face(embedding)
    if duplicate:
        raise HTTPException(
            status_code=400, 
            detail=f"Registration failed: This face is already registered under the name '{duplicate['name']}'."
        )

    user_id = db.add_user(req.name.strip(), embedding)
    return {"success": True, "user_id": user_id, "message": f"{req.name.strip()} registered successfully!"}


@app.post("/api/register-multi")
def register_multi(req: MultiRegisterRequest):
    """Register with multiple captures for more robust matching.
    Accepts 2-5 images, averages their embeddings.
    Uses DeepFace for robust face detection."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if len(req.images) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images are required")
    if len(req.images) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed")

    embeddings = []
    for i, img_b64 in enumerate(req.images):
        try:
            img = decode_image(img_b64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image data in capture {i + 1}")

        result = smart_extract_embedding(img)
        if result is None:
            raise HTTPException(
                status_code=400,
                detail=f"No face detected in capture {i + 1}. Please ensure your face is clearly visible and well-lit."
            )

        embeddings.append(result["embedding"])

    # Average the embeddings for a more robust representation
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    
    # Check for duplicate face
    duplicate = check_duplicate_face(avg_embedding)
    if duplicate:
        raise HTTPException(
            status_code=400, 
            detail=f"Registration failed: This face is already registered under the name '{duplicate['name']}'."
        )

    user_id = db.add_user(req.name.strip(), avg_embedding)
    return {
        "success": True,
        "user_id": user_id,
        "captures_used": len(embeddings),
        "message": f"{req.name.strip()} registered with {len(embeddings)} captures for enhanced accuracy!",
    }


@app.post("/api/liveness-check")
def liveness_check(req: LivenessCheckRequest):
    """Perform blink-based liveness detection + LBP texture analysis.
    Returns whether the person is live (not a photo/video proxy)."""
    if len(req.frames) < 3:
        raise HTTPException(status_code=400, detail="At least 3 frames are required for liveness check")

    try:
        frames = [decode_image(f) for f in req.frames]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid frame data")

    # 1) Blink detection
    blink_count = detect_blinks_in_frames(frames)

    # 2) Texture analysis on the first frame with a detected face
    texture_pass = False
    face_embedding = None
    for frame in frames:
        faces = detect_faces_opencv(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = crop_face_with_padding(frame, x, y, w, h)
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            lbp_var = compute_lbp_variance(gray_face)
            texture_pass = lbp_var > LBP_SPOOF_THRESHOLD
            try:
                face_embedding = get_embedding(face_crop)
            except Exception:
                pass
            break

    blink_pass = blink_count >= MIN_BLINKS_REQUIRED
    liveness_passed = blink_pass and texture_pass

    return {
        "liveness": liveness_passed,
        "blink_detected": blink_pass,
        "blink_count": blink_count,
        "texture_pass": texture_pass,
        "embedding": face_embedding,
        "message": "Liveness verified! ✓" if liveness_passed
                   else "Liveness check failed. "
                        + ("No blink detected. " if not blink_pass else "")
                        + ("Flat surface detected. " if not texture_pass else ""),
    }


@app.post("/api/recognize")
def recognize_faces(req: RecognizeRequest):
    """Recognize faces in a frame and record attendance for matches.
    Uses DeepFace for robust face detection."""
    try:
        img = decode_image(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Use DeepFace to detect all faces and get embeddings
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = DeepFace.represent(
            rgb,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
        if not isinstance(results, list):
            results = [results]
    except Exception:
        # Fallback: try with skip detection (one face assumed)
        result = extract_embedding_fallback(img)
        results = [result] if result else []

    if not results:
        return {"recognized": [], "unknown_count": 0, "face_locations": []}

    # Load all known users
    users = db.get_all_users()
    if not users:
        face_locs = []
        for r in results:
            fa = r.get("facial_area", {})
            x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
            face_locs.append([int(y), int(x + w), int(y + h), int(x)])
        return {"recognized": [], "unknown_count": len(results), "face_locations": face_locs}

    known_embeddings = [u["encoding"] for u in users]
    known_names = [u["name"] for u in users]
    known_ids = [u["id"] for u in users]

    recognized = []
    unknown_count = 0

    for r in results:
        embedding = r.get("embedding")
        fa = r.get("facial_area", {})
        x = fa.get("x", 0)
        y = fa.get("y", 0)
        w = fa.get("w", 0)
        h = fa.get("h", 0)

        if not embedding:
            unknown_count += 1
            continue

        # Compare against all known faces
        distances = [cosine_distance(embedding, ke) for ke in known_embeddings]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]

        if min_dist < VERIFICATION_THRESHOLD:
            name = known_names[min_idx]
            user_id = known_ids[min_idx]
            
            att_id = db.record_attendance(user_id, name, liveness_verified=False)
            recognized.append({
                "name": name,
                "user_id": user_id,
                "confidence": round(1.0 - min_dist, 2),
                "location": [int(y), int(x + w), int(y + h), int(x)],
                "already_recorded": att_id == -1,
                "holiday_blocked": att_id == -2,
            })
        else:
            unknown_count += 1

    face_locs = []
    for r in results:
        fa = r.get("facial_area", {})
        x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
        face_locs.append([int(y), int(x + w), int(y + h), int(x)])

    return {
        "recognized": recognized,
        "unknown_count": unknown_count,
        "face_locations": face_locs,
    }


@app.post("/api/recognize-with-liveness")
def recognize_with_liveness(req: LivenessCheckRequest):
    """Combined endpoint: run liveness check on frame sequence, then recognize & record.
    Only records attendance if liveness passes.
    Uses DeepFace for robust face detection."""
    if len(req.frames) < 3:
        raise HTTPException(status_code=400, detail="At least 3 frames needed")

    try:
        frames = [decode_image(f) for f in req.frames]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid frame data")

    # Liveness check (uses Haar cascade for blink detection — that's fine for eyes)
    blink_count = detect_blinks_in_frames(frames)
    blink_pass = blink_count >= MIN_BLINKS_REQUIRED

    # Find a good frame with a face for texture + recognition using DeepFace
    best_result = None
    best_frame = None
    for frame in frames:
        result = smart_extract_embedding(frame)
        if result is not None:
            best_result = result
            best_frame = frame
            break

    if best_result is None or best_frame is None:
        return {"liveness": False, "recognized": [], "message": "No face detected in frames. Please ensure your face is clearly visible."}

    # Texture check using the facial area from DeepFace
    fa = best_result.get("facial_area", {})
    x = fa.get("x", 0)
    y = fa.get("y", 0)
    w = fa.get("w", best_frame.shape[1])
    h = fa.get("h", best_frame.shape[0])
    face_crop = crop_face_with_padding(best_frame, x, y, w, h)
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lbp_var = compute_lbp_variance(gray_face)
    texture_pass = lbp_var > LBP_SPOOF_THRESHOLD

    liveness_passed = blink_pass and texture_pass

    if not liveness_passed:
        msg = "Liveness failed: "
        if not blink_pass:
            msg += "No blink detected. "
        if not texture_pass:
            msg += "Possible photo/flat surface detected. "
        return {"liveness": False, "recognized": [], "message": msg}

    # Liveness passed — now recognize using the embedding we already have
    embedding = best_result["embedding"]

    users = db.get_all_users()
    if not users:
        return {"liveness": True, "recognized": [], "message": "Liveness passed but no registered users."}

    distances = [cosine_distance(embedding, u["encoding"]) for u in users]
    min_idx = int(np.argmin(distances))
    min_dist = distances[min_idx]

    if min_dist < VERIFICATION_THRESHOLD:
        user = users[min_idx]
        
        att_id = db.record_attendance(user[id], user["name"], liveness_verified=True)
        
        if att_id == -2:
            msg = f"✓ Verified {user['name']}, but attendance blocked (Sunday/Holiday)."
        elif att_id == -1:
            msg = f"✓ Liveness verified. Attendance already recorded for {user['name']}."
        else:
            msg = f"✓ Liveness verified & attendance recorded for {user['name']}!"

        return {
            "liveness": True,
            "recognized": [{
                "name": user["name"],
                "user_id": user["id"],
                "confidence": round(1.0 - min_dist, 2),
                "already_recorded": att_id == -1,
                "holiday_blocked": att_id == -2,
            }],
            "message": msg,
        }
    else:
        return {
            "liveness": True,
            "recognized": [],
            "message": "Liveness passed but face not recognized. Please register first.",
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


@app.get("/api/heatmap")
def get_heatmap():
    """Return weekly attendance heatmap data."""
    return {"heatmap": db.get_weekly_heatmap()}


@app.get("/api/export-csv")
def export_csv(date: str | None = Query(None)):
    """Export attendance records as a downloadable CSV file."""
    csv_data = db.export_attendance_csv(date)
    filename = f"attendance_{date or 'all'}.csv"
    return StreamingResponse(
        io.BytesIO(csv_data.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
