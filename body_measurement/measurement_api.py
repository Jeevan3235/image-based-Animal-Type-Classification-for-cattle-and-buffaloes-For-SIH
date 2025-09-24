from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io

from landmarks_detector import detect_landmarks
from measurement_utils import (
    compute_body_length,
    compute_height_at_withers,
    compute_chest_width,
    compute_rump_angle,
    compute_classification_score
)

app = FastAPI()

def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file))
    image = image.convert("RGB")
    arr = np.array(image)
    # BGR needed for OpenCV
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

@app.post("/measure")
async def measure(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_imagefile(contents)
    landmarks = detect_landmarks(img)
    # compute measures
    m = {}
    m["body_length"] = compute_body_length(landmarks)
    m["height"] = compute_height_at_withers(landmarks)
    m["chest_width"] = compute_chest_width(landmarks)
    m["rump_angle"] = compute_rump_angle(landmarks)
    m["score"] = compute_classification_score(m)
    # build output
    out = {
        "landmarks": landmarks,
        "measurements": m
    }
    return JSONResponse(content=out)
