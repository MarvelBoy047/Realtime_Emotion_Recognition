import streamlit as st
import cv2
from keras.models import load_model
import numpy as np

def main():
    # Load pre-trained Keras model for emotion recognition
    model = load_model('emotion_detection_model.h5')

    # Define emotions
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Load the cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the camera
    cap = cv2.VideoCapture(0)

    # Set Streamlit page configuration
    st.set_page_config(page_title="Realtime_Emotion_Recognition", layout="centered")

    # Display title and description
    st.title("Realtime Emotion Recognition")

    # Create placeholders for buttons and camera feed
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")
    frame_placeholder = st.empty()

    # Variable to track camera status
    camera_running = False

    # Main loop for capturing frames and performing emotion prediction
    while True:
        if start_button:
            camera_running = True
            start_button = False
            stop_button = False
        elif stop_button:
            camera_running = False
            start_button = False
            stop_button = False

        if camera_running:
            # Read frame from the camera
            ret, frame = cap.read()

            if not ret:
                st.write("Unable to capture frame.")
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
                label = emotions[np.argmax(result)]

                # Draw the detected face and emotion label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            frame_placeholder.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not camera_running:
            # Clear the frame placeholder when camera is stopped
            frame_placeholder.empty()
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
