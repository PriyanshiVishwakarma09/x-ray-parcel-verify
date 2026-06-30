# ================= ORIGINAL IMPORTS =================
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import torch
import cv2
import numpy as np
from datetime import datetime
import asyncio
import threading
import sys

sys.path.append("D:/Projects/yolov7/yolov7") #should point to yolov7 folder in your PC

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

# ✅ NEW: YOLOv8 import
from ultralytics import YOLO

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static/my_results", exist_ok=True)

# ================= YOLOv7 MODEL =================
weights = "24_july_detection.pt"
device = select_device('cpu')

print(" Current Device ", device)

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
img_size = check_img_size(640, s=stride)
model.eval()

# ================= ✅ YOLOv8 MODEL =================
iphone_air_model = YOLO("best.pt")

# ================= ORIGINAL CLASS DICT =================
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

conf_thresh = 0.68

# ================= API =================
@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Please upload a valid image file."})

    original_filename = file.filename
    input_image_path = f"static/my_results/{original_filename}"

    with open(input_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img0 = cv2.imread(input_image_path)
        if img0 is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        # ================= YOLOv7 DETECTION =================
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thresh, 0.45)

        detected_classes = set()

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    class_name = object_dict.get(int(cls), "Unknown")
                    detected_classes.add(class_name)

        # ================= ORIGINAL LABEL GROUPS =================
        iphone_related_labels = {"iphone_camera", "iphone_chip", "iphone_charger", "mobile_phone"}
        apple_watch_related_labels = {'watch','watch_chip'}
        apple_tab_related_labels = {'tablet','apple_logo','apple_tab_pcb'}
        iphone_16e_related_labels = {'iphone_chip','mobile_phone','mobile_hardware','mobile_wire'}
        apple_pencil_related_labels = {'pencil','apple_pencil_chip','apple_pencil_back'}
        apple_buds_related_labels = {'buds','airpod_pcb','bus_dot'}
        sony_camera_related_labels = {'camera','lens_shutter'}
        nikon_camera_related_labels = {"camera","shutter","green_grip"}
        ipad_mini_related_labels = {"apple_tab_chip","tablet","apple_logo"}
        one_plus_pencil_related_labels = {'pencil_back','pencil','apple_pencil_chip'}

        # ================= INTERSECTIONS =================
        detected_iphone_labels = detected_classes.intersection(iphone_related_labels)
        detected_apple_watch_labels = detected_classes.intersection(apple_watch_related_labels)
        detected_tab_related_labels = detected_classes.intersection(apple_tab_related_labels)
        detected_iphone_16e_related_labels = detected_classes.intersection(iphone_16e_related_labels)
        detected_pencil_related_labels = detected_classes.intersection(apple_pencil_related_labels)
        detected_buds_related_labels = detected_classes.intersection(apple_buds_related_labels)
        detected_sony_camera_related_labels = detected_classes.intersection(sony_camera_related_labels)
        detected_nikon_camera_relatd_labels = detected_classes.intersection(nikon_camera_related_labels)
        detected_ipad_mini_related_labels = detected_classes.intersection(ipad_mini_related_labels)
        detected_one_plus_related_labels = detected_classes.intersection(one_plus_pencil_related_labels)

        results = []
        iphone_detected = False

        # ================= ORIGINAL LOGIC (UNCHANGED) =================

        if detected_iphone_16e_related_labels:
            if len(detected_iphone_16e_related_labels) >= 3 and not iphone_detected:
                results.append("iPhone 16 e Detected")
                iphone_detected = True

        if detected_iphone_labels:
            if len(detected_iphone_labels) >= 3:
                results.append("iPhone Detected")
                iphone_detected = True
            elif len(detected_iphone_labels) <= 2 and not iphone_detected:
                results.append("Not an iPhone")
                iphone_detected = True

        if detected_classes & {"watch", "watch_sensor", "watch_pcb"}:
            if {"watch", "watch_sensor", "watch_pcb"}.issubset(detected_classes):
                results.append("Not an Apple Watch")
            elif detected_classes & {"watch", "watch_chip"} == {"watch", "watch_chip"}:
                results.append("Apple Watch Detected")
            elif len(detected_classes & {"watch", "watch_chip", "watch_sensor", "watch_pcb"}) == 1:
                results.append("Not an Apple Watch")

        if detected_tab_related_labels:
            if len(detected_tab_related_labels) >= 2:
                results.append("iPad Detected")
            elif 1 <= len(detected_tab_related_labels) < 2 or "tablet" in detected_classes:
                results.append("Not an iPad")

        if detected_ipad_mini_related_labels:
            if len(detected_ipad_mini_related_labels) >= 2:
                results.append("iPad Detected")
            elif 1 <= len(detected_ipad_mini_related_labels) < 2 or "tablet" in detected_classes:
                results.append("Not an iPad")

        apple_pencil_detected = False

        if detected_one_plus_related_labels:
            if len(detected_one_plus_related_labels) >= 3:
                results.append("Not an Apple pencil")
                apple_pencil_detected = True

        if apple_pencil_detected != True:
            if detected_pencil_related_labels:
                if len(detected_pencil_related_labels) >= 2:
                    results.append("Apple pencil Detected")
                elif len(detected_pencil_related_labels) == 1:
                    results.append("Not an Apple pencil")

        camera_detected = False

        if detected_nikon_camera_relatd_labels:
            if len(detected_nikon_camera_relatd_labels) >= 2:
                results.append("Nikon Camera Detected")
                camera_detected = True

        if detected_sony_camera_related_labels:
            if len(detected_sony_camera_related_labels) >= 2:
                results.append("Sony Camera Detected")
                camera_detected = True
            elif not camera_detected:
                results.append("Camera Detected")
                camera_detected = True

        if detected_buds_related_labels:
            if len(detected_buds_related_labels) >= 2:
                results.append("Apple Airpods Detected")
            elif len(detected_buds_related_labels) == 1:
                results.append("Not an Apple Airpods")

        # ================= ✅ NEW: YOLOv8 iPhone Air =================
        air_results = iphone_air_model(img0)
        iphone_air_detected = set()

        if "mobile_phone" in detected_classes:

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
                
        # ================= FINAL =================
        if not results:
            results.append("Uncertain ITEM")

        classification = ", ".join(results)
        text_color = (0,255,0) if any("Detected" in r for r in results) else (0,0,255)

        cv2.putText(img0, classification, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.93, text_color, 3)

        output_filename = f"detection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{original_filename}"
        output_path = f"static/my_results/{output_filename}"
        cv2.imwrite(output_path, img0)

        processed_image_url = f"{request.base_url}static/my_results/{output_filename}"

        return JSONResponse({
            "classification": classification,
            "detected_classes": list(detected_classes),
            "iphone_air_classes": list(iphone_air_detected),
            "result_image": processed_image_url
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000)