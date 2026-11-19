"""
Hugging Face Spaces Entrypoint — Gradio Web Interface for X-Ray Parcel Verify
Runs on Free 16GB RAM CPU hardware on Hugging Face Spaces.
"""

import os
import sys
import cv2
import numpy as np
import torch
import gradio as gr

# Ensure YOLOv7 repository is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV7_PATH = os.getenv("YOLOV7_PATH", os.path.join(BASE_DIR, "yolov7"))

# Clone YOLOv7 if not present (for Hugging Face Spaces automatic clone)
if not os.path.isdir(YOLOV7_PATH):
    print("[INFO] Cloning YOLOv7 repository for Hugging Face Spaces...")
    os.system("git clone --depth 1 https://github.com/WongKinYiu/yolov7.git " + YOLOV7_PATH)

if os.path.isdir(YOLOV7_PATH) and YOLOV7_PATH not in sys.path:
    sys.path.insert(0, YOLOV7_PATH)

from ultralytics import YOLO
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Model weights
YOLOV7_WEIGHTS = os.path.join(BASE_DIR, "24_july_detection.pt")
YOLOV8_WEIGHTS = os.path.join(BASE_DIR, "best.pt")

# Device & Models
device = select_device('cpu')
print("[INFO] Loading YOLOv7 model on Hugging Face...")
model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
stride = int(model.stride.max())
img_size = check_img_size(640, s=stride)
model.eval()

print("[INFO] Loading YOLOv8 model on Hugging Face...")
iphone_air_model = YOLO(YOLOV8_WEIGHTS)

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

from client_code import run_heuristics

def inspect_parcel(input_image):
    if input_image is None:
        return None, "No image uploaded.", {}

    # input_image from Gradio is RGB numpy array
    img0 = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # YOLOv7 preprocessing
    img = letterbox(img0, img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.68, 0.45)

    detected_classes = set()
    components_dict = {}

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                class_name = object_dict.get(int(cls), "Unknown")
                detected_classes.add(class_name)
                c_float = float(conf)
                components_dict[class_name] = round(c_float, 3)
                x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(img0, f"{class_name} {c_float:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

    # Run heuristics
    results = run_heuristics(detected_classes)

    # YOLOv8 iPhone Air check
    if "mobile_phone" in detected_classes:
        air_results = iphone_air_model(img0)
        iphone_air_detected = set()
        for r in air_results:
            for box in r.boxes:
                if float(box.conf[0]) >= 0.90:
                    cls_id = int(box.cls[0])
                    iphone_air_detected.add(iphone_air_model.names[cls_id])

        if len(iphone_air_detected) >= 3:
            results.append("iPhone Air Detected")
        elif len(iphone_air_detected) <= 1:
            results.append("Not an iPhone Air")
        else:
            results.append("iPhone Air Inconclusive")

    if not results:
        results.append("Uncertain ITEM")

    unique_results = list(dict.fromkeys(results))
    classification = ", ".join(unique_results)
    text_color = (0, 230, 0) if any("Detected" in r for r in unique_results) else (0, 0, 230)

    # Render header banner
    h, w = img0.shape[:2]
    banner_h = max(60, int(len(unique_results) * 28 + 20))
    cv2.rectangle(img0, (0, 0), (w, banner_h), (25, 25, 25), -1)

    y_offset = 32
    for line in unique_results:
        cv2.putText(img0, f"• {line}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
        y_offset += 28

    # Convert back to RGB for Gradio display
    out_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    return out_rgb, classification, components_dict

# Gradio Interface
demo = gr.Interface(
    fn=inspect_parcel,
    inputs=gr.Image(type="numpy", label="Upload Parcel X-Ray Scan"),
    outputs=[
        gr.Image(type="numpy", label="Annotated X-Ray Scan (Bounding Boxes)"),
        gr.Textbox(label="AI Verification Verdict", interactive=False),
        gr.Label(label="Detected Hardware Components & Confidences")
    ],
    title="📦 X-Ray Parcel Verify — AI Security Terminal",
    description="Upload an X-ray parcel scanner image to detect internal electronics components and verify genuine vs. counterfeit devices.",
    theme="dark"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

