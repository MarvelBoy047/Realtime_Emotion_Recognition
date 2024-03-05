# Realtime Emotion Recognition

This Python script performs real-time emotion recognition using a laptop camera. It detects faces in the video stream and predicts the corresponding emotion for each detected face.

## Requirements

- Python 3.11
- OpenCV
- Keras
- NumPy
- Flask

## Installation

You can install the required Python packages using pip:

```bash
pip install opencv-python keras numpy Flask
```

## Usage

1. Make sure your laptop camera is properly connected and accessible.
2. Run the `app.py` script.
3. Open a web browser and navigate to `http://127.0.0.1:8080`.
4. Click the "Start" button to begin emotion detection.
5. Click the "Stop" button to stop emotion detection.
6. The webpage will display the camera feed with emotion labels drawn on detected faces.
7. Press 'q' in the terminal where the script is running to exit the application.

## Model

The emotion detection model (`emotion_detection_model.h5`) used in this script is pre-trained and loaded using Keras. If you need to train your own model or download a different pre-trained model, make sure to replace the existing model file and update the code accordingly.

## Improvements

1. **Performance Optimization**: Implement optimizations like face tracking to reduce redundant face detection computations.
2. **User Interface**: Enhance the user interface by adding controls for starting/stopping the camera, adjusting detection parameters, or displaying additional information.
3. **Multi-threading**: Utilize multi-threading to improve performance and responsiveness, especially on systems with multiple CPU cores.
4. **Model Fine-tuning**: Fine-tune the pre-trained model using additional data for better accuracy, or explore other deep learning architectures.
5. **Error Handling**: Implement robust error handling to gracefully handle exceptions and edge cases.
6. **Dynamic Graph**: Add a dynamic graph beside the camera feed to show the percentage of detected emotions in real-time.
7. **Deployment**: Package the script into an executable or deploy it as a web application for easier distribution and access.
