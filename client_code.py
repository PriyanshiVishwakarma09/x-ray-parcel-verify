"""
X-Ray Parcel Verify — FastAPI Object Detection & Component Verification Service
"""

import os
import sys
import uuid
import shutil
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ================= CONFIGURATION & ENVIRONMENT =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Portable YOLOv7 path resolution (Container /app/yolov7, local ./yolov7, or env var)
YOLOV7_PATH = os.getenv("YOLOV7_PATH", "/app/yolov7")
if not os.path.isdir(YOLOV7_PATH):
    local_yolov7 = os.path.join(BASE_DIR, "yolov7")
    if os.path.isdir(local_yolov7):
        YOLOV7_PATH = local_yolov7

if os.path.isdir(YOLOV7_PATH) and YOLOV7_PATH not in sys.path:
    sys.path.insert(0, YOLOV7_PATH)

# Optional dynamic import of YOLOv7 modules
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords, check_img_size
    from utils.torch_utils import select_device
    from utils.datasets import letterbox
    YOLOV7_AVAILABLE = True
except ImportError as e:
    YOLOV7_AVAILABLE = False
    print(f"[WARNING] YOLOv7 modules could not be imported from '{YOLOV7_PATH}': {e}")

# Device & Performance configurations
DEVICE_NAME = os.getenv("DEVICE", "cpu")
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.68"))
IOU_THRESH = float(os.getenv("IOU_THRESH", "0.45"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))

# Limit CPU threads to reduce RAM footprint on cloud free-tiers (Render/HF)
if DEVICE_NAME.lower() == "cpu":
    torch.set_num_threads(TORCH_THREADS)

# Storage paths
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(BASE_DIR, "static", "my_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Weight paths
YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", os.path.join(BASE_DIR, "24_july_detection.pt"))
YOLOV8_WEIGHTS = os.getenv("YOLOV8_WEIGHTS", os.path.join(BASE_DIR, "best.pt"))

# ================= FASTAPI APP INITIALIZATION =================
app = FastAPI(
    title="X-Ray Parcel Verify API",
    description="Machine Learning Computer Vision API for verifying electronic parcels from X-Ray scans.",
    version="2.0.0"
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static directory for results
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ================= MODEL LOADING =================
model = None
stride = 32
img_size = 640
device = None

if YOLOV7_AVAILABLE and os.path.isfile(YOLOV7_WEIGHTS):
    try:
        device = select_device(DEVICE_NAME)
        print(f"[INFO] Loading YOLOv7 model on device: {device} from {YOLOV7_WEIGHTS}")
        model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
        stride = int(model.stride.max())
        img_size = check_img_size(640, s=stride)
        model.eval()
        print("[INFO] YOLOv7 model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv7 model: {e}")
        model = None
else:
    print(f"[WARNING] YOLOv7 model weights not found or modules unavailable at {YOLOV7_WEIGHTS}")

# Load YOLOv8 model for iPhone Air check
iphone_air_model = None
if os.path.isfile(YOLOV8_WEIGHTS):
    try:
        print(f"[INFO] Loading YOLOv8 model from {YOLOV8_WEIGHTS}")
        iphone_air_model = YOLO(YOLOV8_WEIGHTS)
        print("[INFO] YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8 model: {e}")
        iphone_air_model = None

# ================= CLASS DICTIONARY =================
object_dict = {
    0: "iphone_camera", 1: "iphone_chip", 2: "iphone_charger", 3: "mobile_phone",
    4: "watch", 5: "watch_chip", 6: "watch_pcb", 7: "watch_sensor",
    8: "tablet", 9: "apple_logo", 10: "apple_tab_pcb",
    11: "buds", 12: "buds_pcb", 13: "airpod_pcb",
    14: "pencil", 15: "pencil_back", 16: "apple_pencil_chip",
    17: "mobile_hardware", 18: "mobile_wire",
    19: "camera", 20: "shutter", 21: "lens_shutter",
    22: "green_grip", 23: "apple_pencil_back",
    24: "apple_tab_chip", 25: "bus_dot"
}

# ================= HEALTH & ROOT ENDPOINTS =================
@app.get("/", tags=["General"])
def root_status():
    return {
        "service": "X-Ray Parcel Verify API",
        "status": "online",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy" if (model is not None and iphone_air_model is not None) else "degraded",
        "yolov7_loaded": model is not None,
        "yolov8_loaded": iphone_air_model is not None,
        "device": str(device) if device is not None else DEVICE_NAME,
        "yolov7_path": YOLOV7_PATH
    }

# ================= CLASSIFICATION HEURISTICS =================
def run_heuristics(detected_classes: set) -> List[str]:
    """
    Applies domain-specific component correlation heuristics
    to determine genuine vs counterfeit/mismatched products.
    """
    iphone_related_labels = {"iphone_camera", "iphone_chip", "iphone_charger", "mobile_phone"}
    apple_watch_related_labels = {'watch', 'watch_chip'}
    apple_tab_related_labels = {'tablet', 'apple_logo', 'apple_tab_pcb'}
    iphone_16e_related_labels = {'iphone_chip', 'mobile_phone', 'mobile_hardware', 'mobile_wire'}
    apple_pencil_related_labels = {'pencil', 'apple_pencil_chip', 'apple_pencil_back'}
    apple_buds_related_labels = {'buds', 'airpod_pcb', 'bus_dot'}
    sony_camera_related_labels = {'camera', 'lens_shutter'}
    nikon_camera_related_labels = {"camera", "shutter", "green_grip"}
    ipad_mini_related_labels = {"apple_tab_chip", "tablet", "apple_logo"}
    one_plus_pencil_related_labels = {'pencil_back', 'pencil', 'apple_pencil_chip'}

    detected_iphone_labels = detected_classes.intersection(iphone_related_labels)
    detected_apple_watch_labels = detected_classes.intersection(apple_watch_related_labels)
    detected_tab_related_labels = detected_classes.intersection(apple_tab_related_labels)
    detected_iphone_16e_related_labels = detected_classes.intersection(iphone_16e_related_labels)
    detected_pencil_related_labels = detected_classes.intersection(apple_pencil_related_labels)
    detected_buds_related_labels = detected_classes.intersection(apple_buds_related_labels)
    detected_sony_camera_related_labels = detected_classes.intersection(sony_camera_related_labels)
    detected_nikon_camera_related_labels = detected_classes.intersection(nikon_camera_related_labels)
    detected_ipad_mini_related_labels = detected_classes.intersection(ipad_mini_related_labels)
    detected_one_plus_related_labels = detected_classes.intersection(one_plus_pencil_related_labels)

    results = []
    iphone_detected = False

    # iPhone 16e check
    if detected_iphone_16e_related_labels and len(detected_iphone_16e_related_labels) >= 3:
        results.append("iPhone 16 e Detected")
        iphone_detected = True

    # Standard iPhone check
    if detected_iphone_labels:
        if len(detected_iphone_labels) >= 3 and not iphone_detected:
            results.append("iPhone Detected")
            iphone_detected = True
        elif len(detected_iphone_labels) <= 2 and not iphone_detected:
            results.append("Not an iPhone")
            iphone_detected = True

    # Apple Watch check
    if detected_classes & {"watch", "watch_sensor", "watch_pcb"}:
        if {"watch", "watch_sensor", "watch_pcb"}.issubset(detected_classes):
            results.append("Not an Apple Watch")
        elif detected_classes & {"watch", "watch_chip"} == {"watch", "watch_chip"}:
            results.append("Apple Watch Detected")
        elif len(detected_classes & {"watch", "watch_chip", "watch_sensor", "watch_pcb"}) == 1:
            results.append("Not an Apple Watch")

    # iPad check (Deduplicated logic to prevent duplicate labels)
    ipad_evaluated = False
    if detected_tab_related_labels and len(detected_tab_related_labels) >= 2:
        results.append("iPad Detected")
        ipad_evaluated = True
    elif detected_ipad_mini_related_labels and len(detected_ipad_mini_related_labels) >= 2 and not ipad_evaluated:
        results.append("iPad Detected")
        ipad_evaluated = True
    elif ("tablet" in detected_classes or (detected_tab_related_labels and len(detected_tab_related_labels) == 1)) and not ipad_evaluated:
        results.append("Not an iPad")
        ipad_evaluated = True

    # Pencil check
    apple_pencil_detected = False
    if detected_one_plus_related_labels and len(detected_one_plus_related_labels) >= 3:
        results.append("Not an Apple pencil")
        apple_pencil_detected = True

    if not apple_pencil_detected and detected_pencil_related_labels:
        if len(detected_pencil_related_labels) >= 2:
            results.append("Apple pencil Detected")
        elif len(detected_pencil_related_labels) == 1:
            results.append("Not an Apple pencil")

    # Camera check
    camera_detected = False
    if detected_nikon_camera_related_labels and len(detected_nikon_camera_related_labels) >= 2:
        results.append("Nikon Camera Detected")
        camera_detected = True

    if detected_sony_camera_related_labels and len(detected_sony_camera_related_labels) >= 2:
        results.append("Sony Camera Detected")
        camera_detected = True
    elif not camera_detected and ("camera" in detected_classes):
        results.append("Camera Detected")
        camera_detected = True

    # AirPods check
    if detected_buds_related_labels:
        if len(detected_buds_related_labels) >= 2:
            results.append("Apple Airpods Detected")
        elif len(detected_buds_related_labels) == 1:
            results.append("Not an Apple Airpods")

    return results

# ================= INFERENCE ENDPOINT =================
# Defined as synchronous def so FastAPI offloads CPU-bound ML work to worker threadpool
@app.post("/upload", tags=["Detection"])
def upload_image(request: Request, file: UploadFile = File(...)):
    """
    Upload an X-ray parcel image to run computer vision verification.
    Returns detected classes, product verification results, bounding boxes, and processed image URL.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection model is currently not loaded. Check server logs."
        )

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload an image file (JPEG, PNG, etc.)."
        )

    # Secure unique filename generation
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        file_ext = ".jpg"

    unique_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_filename = f"upload_{timestamp}_{unique_id}{file_ext}"
    input_image_path = os.path.join(RESULTS_DIR, temp_filename)

    # Save file with size limit enforcement
    try:
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        total_size = 0
        with open(input_image_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 64):
                total_size += len(chunk)
                if total_size > max_bytes:
                    buffer.close()
                    if os.path.exists(input_image_path):
                        os.remove(input_image_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File upload failed: {e}")

    try:
        img0 = cv2.imread(input_image_path)
        if img0 is None:
            if os.path.exists(input_image_path):
                os.remove(input_image_path)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted or unreadable image file.")

        # ================= YOLOv7 DETECTION =================
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)

        detected_classes = set()
        detections_list: List[Dict[str, Any]] = []

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    class_name = object_dict.get(int(cls), "Unknown")
                    detected_classes.add(class_name)
                    c_float = float(conf)
                    x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                    detections_list.append({
                        "class": class_name,
                        "confidence": round(c_float, 4),
                        "box": [x1, y1, x2, y2]
                    })
                    # Draw detection bounding boxes with labels
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 120, 0), 2)
                    label_str = f"{class_name} {c_float:.2f}"
                    cv2.putText(img0, label_str, (x1, max(y1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

        # ================= RULE-BASED HEURISTICS =================
        results = run_heuristics(detected_classes)

        # ================= OPTIMIZED YOLOv8 IPHONE AIR CHECK =================
        iphone_air_detected = set()
        # Only run YOLOv8 if 'mobile_phone' is detected and model is loaded
        if "mobile_phone" in detected_classes and iphone_air_model is not None:
            air_results = iphone_air_model(img0)
            for r in air_results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= 0.90:
                        cls_id = int(box.cls[0])
                        label = iphone_air_model.names[cls_id]
                        iphone_air_detected.add(label)

            if len(iphone_air_detected) >= 3:
                results.append("iPhone Air Detected")
            elif len(iphone_air_detected) <= 1:
                results.append("Not an iPhone Air")
            else:
                results.append("iPhone Air Inconclusive")

        # Fallback if no specific rule matched
        if not results:
            results.append("Uncertain ITEM")

        # Deduplicate results preserving order
        unique_results = list(dict.fromkeys(results))
        classification = ", ".join(unique_results)
        text_color = (0, 230, 0) if any("Detected" in r for r in unique_results) else (0, 0, 230)

        # ================= RESULT BANNER RENDERING =================
        # Draw background header bar for high readability
        h, w = img0.shape[:2]
        banner_h = max(60, int(len(unique_results) * 28 + 20))
        cv2.rectangle(img0, (0, 0), (w, banner_h), (25, 25, 25), -1)

        y_offset = 32
        for line in unique_results:
            cv2.putText(img0, f"• {line}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            y_offset += 28

        # Save processed output image
        output_filename = f"detection_{timestamp}_{unique_id}.jpg"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        cv2.imwrite(output_path, img0)

        # Clean up temporary upload if separate
        if os.path.exists(input_image_path) and input_image_path != output_path:
            try:
                os.remove(input_image_path)
            except Exception:
                pass

        # Cleanup memory for cloud instances
        gc.collect()

        processed_image_url = f"{str(request.base_url).rstrip('/')}/static/my_results/{output_filename}"

        return JSONResponse({
            "status": "success",
            "classification": classification,
            "results": unique_results,
            "detected_classes": sorted(list(detected_classes)),
            "iphone_air_classes": sorted(list(iphone_air_detected)),
            "detections_count": len(detections_list),
            "detections": detections_list,
            "result_image": processed_image_url
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        )


# ================= RUN SERVER =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app=app, host=host, port=port)
