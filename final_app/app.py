"""
Open the browser to: http://127.0.0.1:5000
"""
import io
import time
import uuid
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename

WEIGHTS_PATH = Path(__file__).resolve().parent / "best.pt"
model = YOLO(str(WEIGHTS_PATH)) if WEIGHTS_PATH.exists() else None
CLASS_NAMES = model.names if model else {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB Max upload limit
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}

video_sessions = {}
video_sessions_lock = threading.Lock()


def allowed_video(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT


PALETTE = [
    (232, 89, 11), (11, 158, 232), (196, 30, 58), (79, 209, 122),
    (242, 193, 78), (140, 151, 164), (192, 57, 39), (11, 89, 232),
]


def get_color(cls_id: int):
    return PALETTE[cls_id % len(PALETTE)]


def draw_detections(frame: np.ndarray, result) -> tuple:
    """Draw bounding boxes on frame and return detection details."""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        color = get_color(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        detections.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2],
        })
    return frame, detections


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None,
                           class_names=list(CLASS_NAMES.values()))


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Weights file best.pt was not found alongside app.py"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))
    high_accuracy = request.form.get("high_accuracy", "false") == "true"

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    t0 = time.time()
    result = model.predict(frame, conf=conf, iou=iou, augment=high_accuracy, verbose=False)[0]
    infer_ms = round((time.time() - t0) * 1000, 1)

    annotated, detections = draw_detections(frame.copy(), result)

    ok, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "detections": detections,
        "count": len(detections),
        "inference_ms": infer_ms,
        "image": f"data:image/jpeg;base64,{annotated_b64}",
    })


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Receives video file, saves to disk, and returns session_id for streaming."""
    if model is None:
        return jsonify({"error": "Weights file best.pt was not found alongside app.py"}), 400

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_video(file.filename):
        return jsonify({"error": "Unsupported video format (mp4, avi, mov, mkv, webm)"}), 400

    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))
    high_accuracy = request.form.get("high_accuracy", "false") == "true"

    session_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / f"{session_id}_{safe_name}"
    file.save(saved_path)

    cap_check = cv2.VideoCapture(str(saved_path))
    if not cap_check.isOpened():
        cap_check.release()
        saved_path.unlink(missing_ok=True)
        return jsonify({"error": "Failed to open video file. It may be corrupted or unsupported."}), 400

    total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_check.get(cv2.CAP_PROP_FPS) or 25
    cap_check.release()

    with video_sessions_lock:
        video_sessions[session_id] = {
            "path": str(saved_path),
            "conf": conf,
            "iou": iou,
            "high_accuracy": high_accuracy,
            "total_frames": total_frames,
            "fps": fps,
            "processed_frames": 0,
            "detections_total": 0,
            "done": False,
        }

    return jsonify({
        "session_id": session_id,
        "total_frames": total_frames,
        "fps": round(fps, 1),
    })


def gen_video_stream(session_id: str):
    """Generates MJPEG frames, processing the video frame-by-frame with YOLOv8."""
    with video_sessions_lock:
        session = video_sessions.get(session_id)
    if session is None:
        return

    cap = cv2.VideoCapture(session["path"])
    if not cap.isOpened():
        return

    conf = session["conf"]
    iou = session["iou"]
    high_accuracy = session["high_accuracy"]

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            detections_count = 0
            if model is not None:
                result = model.predict(frame, conf=conf, iou=iou,
                                        augment=high_accuracy, verbose=False)[0]
                frame, dets = draw_detections(frame, result)
                detections_count = len(dets)

            with video_sessions_lock:
                s = video_sessions.get(session_id)
                if s is not None:
                    s["processed_frames"] += 1
                    s["detections_total"] += detections_count

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()
        with video_sessions_lock:
            s = video_sessions.get(session_id)
            if s is not None:
                s["done"] = True
        # Clean up temporary uploaded file after stream finishes
        try:
            Path(session["path"]).unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/video_feed/<session_id>")
def video_feed(session_id):
    with video_sessions_lock:
        exists = session_id in video_sessions
    if not exists:
        return jsonify({"error": "Video session not found or expired"}), 404
    return Response(gen_video_stream(session_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_progress/<session_id>")
def video_progress(session_id):
    with video_sessions_lock:
        s = video_sessions.get(session_id)
    if s is None:
        return jsonify({"error": "Video session not found"}), 404
    return jsonify({
        "processed_frames": s["processed_frames"],
        "total_frames": s["total_frames"],
        "detections_total": s["detections_total"],
        "done": s["done"],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None,
                    "classes": list(CLASS_NAMES.values())})


if __name__ == "__main__":
    if model is None:
        print(f"⚠️  Warning: Weights file not found at {WEIGHTS_PATH}")
        print("    Place best.pt next to app.py or update WEIGHTS_PATH.")
    else:
        print(f"✅ Model loaded successfully. Classes: {list(CLASS_NAMES.values())}")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)