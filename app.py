from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from keras.models import load_model

app = FastAPI()

# Mounting the static directory containing HTML, CSS, and JS files
app.mount("/", StaticFiles(directory="static"), name="static")

class EmotionPrediction(BaseModel):
    emotion: str
    percentage: float

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr_color
