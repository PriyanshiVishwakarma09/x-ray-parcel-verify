# X-Ray Parcel Verify

A FastAPI-based computer vision API that analyzes uploaded X-ray images and classifies whether certain consumer electronics/components are present.  
The project uses:

- **YOLOv7** for the main object detection pipeline
- **YOLOv8** for an additional iPhone Air-specific detection flow
- **OpenCV** and **PyTorch** for image preprocessing and inference

The API returns a text classification, the detected object labels, and a processed result image with the classification drawn on top.

---

## Features

- Upload an image through a REST API
- Detect multiple object/component classes from X-ray images
- Apply custom grouping logic to infer product-level results such as:
  - iPhone Detected / Not an iPhone
  - Apple Watch Detected / Not an Apple Watch
  - iPad Detected / Not an iPad
  - Apple Pencil Detected / Not an Apple pencil
  - Apple Airpods Detected / Not an Apple Airpods
  - Nikon Camera Detected
  - Sony Camera Detected
  - iPhone Air Detected / Not an iPhone Air
- Save processed images to a static results folder
- Return JSON response with:
  - classification text
  - detected classes
  - iPhone Air classes
  - result image URL

---

## How It Works

1. The uploaded image is saved to `static/my_results/`
2. YOLOv7 runs detection on the image
3. Detected classes are mapped to custom object names
4. The detected labels are compared against predefined label groups
5. A final classification string is generated
6. YOLOv8 is used as an extra check for iPhone Air-related labels
7. The classification is drawn on the output image
8. The processed image is saved and returned in the API response

---

## Repository Structure

- `client_code.py` — main FastAPI application and detection logic
- `24_july_detection.pt` — YOLOv7 model weights
- `best.pt` — YOLOv8 model weights
- `requirements.txt` — Python dependencies
- `static/` — served files and detection results

---

## Requirements

This project is built for Python and requires:

- Python 3.9+
- PyTorch
- OpenCV
- FastAPI
- Ultralytics
- YOLOv7 source code available locally

> Important: `client_code.py` currently uses a hardcoded local path for YOLOv7:
>
> ```python
> sys.path.append("D:/Projects/yolov7/yolov7")
> ```
>
> You will need to update this path to match your own YOLOv7 installation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/TheBadassProgrammer/x-ray-parcel-verify.git
cd x-ray-parcel-verify
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install any extra dependencies required by your YOLOv7 setup if they are not already included.

---

## YOLOv7 Setup

This project depends on YOLOv7 utility modules such as:

- `models.experimental`
- `utils.general`
- `utils.torch_utils`
- `utils.datasets`

Make sure the YOLOv7 repository is available on your machine and that this line in `client_code.py` points to it:

```python
sys.path.append("D:/Projects/yolov7/yolov7")
```

Example:

```python
sys.path.append("/path/to/yolov7")
```

---

## Running the Server

Start the FastAPI app with:

```bash
python client_code.py
```

The server will run on:

```text
http://0.0.0.0:8000
```

---

## API Endpoint

### `POST /upload`

Upload an image file for detection and classification.

#### Request
- `file`: image file upload

#### Response Example

```json
{
  "classification": "iPhone Detected, Apple Watch Detected",
  "detected_classes": ["mobile_phone", "iphone_chip"],
  "iphone_air_classes": [],
  "result_image": "http://127.0.0.1:8000/static/my_results/detection_2026-07-05_12-00-00_example.jpg"
}
```

---

## Example Usage

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@sample.jpg"
```

### Python

```python
import requests

url = "http://127.0.0.1:8000/upload"

with open("sample.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

---

## Output Files

Processed images are saved in:

```text
static/my_results/
```

Each output image contains the classification text rendered on the top-left corner.

---

## Notes

- The confidence threshold for YOLOv7 detections is currently set to `0.68`
- The YOLOv8 iPhone Air check only runs when `mobile_phone` is detected by YOLOv7
- The logic is rule-based and depends on combinations of detected classes
- Some category names in the code reflect custom training labels and device-specific heuristics

---

## Limitations

- The project expects local access to both model weight files
- YOLOv7 dependency path is hardcoded and must be updated manually
- Classification logic is heuristic-based and may need tuning for better accuracy
- No frontend is included in the repository
- No Dockerfile or deployment configuration is currently provided

---

## Possible Improvements

- Replace the hardcoded YOLOv7 path with a configurable environment variable
- Add a Docker setup for reproducible deployment
- Add a simple HTML upload form or frontend client
- Add request validation and file-size limits
- Add logging and structured error handling
- Document label meanings and expected detection combinations
- Add example images and sample outputs

---

## License

No license file was found in the repository.  
Add one if you want to define usage and redistribution terms.

---

## Author

Created by **TheBadassProgrammer**