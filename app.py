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


def predict_emotion(image):
    model = load_model('emotion_detection_model.h5')
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    roi_gray = gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray / 255.0
    roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
    result = model.predict(roi_gray)
    label = emotions[np.argmax(result)]
    percentage = float(result[0][np.argmax(result)])
    return label, percentage

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = predict_emotion(img)
    if result:
        emotion, percentage = result
        return EmotionPrediction(emotion=emotion, percentage=percentage)
    else:
        raise HTTPException(status_code=404, detail="No face detected in the image")

@app.get("/")
async def read_root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
