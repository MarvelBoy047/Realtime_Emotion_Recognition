import streamlit as st
import cv2
from keras.models import load_model
import numpy as np

# Parameters for customization
label_font_size = 0.5    # Font size of the emotion labels on the graph
bar_spacing = 10         # Spacing between adjacent bars on the graph
graph_width_ratio = 0.95 # Ratio of the width of the blank space beside the camera window

# Colors for emotions
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral = [(0, 0, 255), (0, 255, 255), (255, 0, 0),
                                                       (0, 255, 0), (255, 0, 255), (255, 255, 0),
                                                       (128, 128, 128)]

def draw_emotion_graph(frame, emotions, percentages, colors, graph_width_ratio):
    height, width, _ = frame.shape
    graph_width = int(width * graph_width_ratio)
    graph_area = np.zeros((height, graph_width, 3), dtype=np.uint8)  # Create an area for the graph
    bar_width = int((graph_width - (len(emotions) - 1) * bar_spacing) / len(emotions))

    for i, (emotion, percentage, color) in enumerate(zip(emotions, percentages, colors)):
        bar_height = int(percentage * height)
        start_point = (i * (bar_width + bar_spacing), height - bar_height)
        end_point = ((i + 1) * bar_width + i * bar_spacing - 1, height)
        cv2.rectangle(graph_area, start_point, end_point, color, -1)
        cv2.putText(graph_area, f"{emotion}: {percentage:.2f}", (i * (bar_width + bar_spacing), height - bar_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, label_font_size, (255, 255, 255), 1, cv2.LINE_AA)

    # Combine the graph area with the frame
    frame_with_graph = np.concatenate((frame, graph_area), axis=1)
    return frame_with_graph

def main():
    # Load pre-trained Keras model for emotion recognition
    model = load_model('emotion_detection_model.h5')

    # Define emotions and their corresponding colors
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Specify custom colors for each emotion (BGR format)
    colors = [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]

    # Load the cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the camera
    cap = cv2.VideoCapture(0)

    # Set Streamlit page configuration
    st.set_page_config(page_title="Real-time Emotion Prediction", layout="wide")

    # Display title and description
    st.title("Real-time Emotion Prediction")
    
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

            # Initialize percentages array to zeros if no faces are detected
            if len(faces) == 0:
                percentages = [0.0] * len(emotions)

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

                # Get the emotion label and percentages
                label = emotions[np.argmax(result)]
                percentages = result.flatten()

                # Draw the detected face and emotion label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw the emotion graph beside the camera window
            frame_with_graph = draw_emotion_graph(frame, emotions, percentages, colors, graph_width_ratio)

            # Display the frame with the graph
            frame_placeholder.image(frame_with_graph, channels="BGR")

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
