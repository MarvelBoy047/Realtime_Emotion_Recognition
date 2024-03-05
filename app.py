from flask import Flask, render_template, Response, jsonify, request
import cv2
from keras.models import load_model
import numpy as np
import threading

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained Keras model for emotion recognition
model = load_model('emotion_detection_model.h5')

# Define emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variable to store the detected emotion
detected_emotion = "Unknown"

# Variable to control emotion detection
detect_emotion_flag = False

# Function to perform emotion detection on webcam frames
def detect_emotion():
    global detected_emotion
    camera = cv2.VideoCapture(0)

    while detect_emotion_flag:
        success, frame = camera.read()
        if not success:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract the face area
            roi_gray = gray[y:y + h, x:x + w]

            # Resize the face area to match the input size of the model
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Normalize the pixels
            roi_gray = roi_gray / 255.0

            # Reshape the image to match the input shape of the model
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

            # Predict the emotion
            result = model.predict(roi_gray)

            # Get the emotion label
            detected_emotion = emotions[np.argmax(result)]

            # Draw the detected face and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame as JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

# Function to start emotion detection
def start_detection():
    global detect_emotion_flag
    detect_emotion_flag = True

# Function to stop emotion detection
def stop_detection():
    global detect_emotion_flag
    detect_emotion_flag = False

# Route to index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for streaming webcam frames
@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get detected emotion
@app.route('/detected_emotion')
def get_detected_emotion():
    return jsonify({'emotion': detected_emotion})

# Route to start emotion detection
@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    start_detection()
    return jsonify({'message': 'Emotion detection started.'})

# Route to stop emotion detection
@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    stop_detection()
    return jsonify({'message': 'Emotion detection stopped.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
