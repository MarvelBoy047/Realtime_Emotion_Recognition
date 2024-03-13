const video = document.getElementById('video');
const emotionLabel = document.getElementById('emotion-label');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
let cameraStream;

async function setupCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = cameraStream;
    } catch (err) {
        console.error('Error accessing camera:', err);
    }
}

async function predictEmotion() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve));
    const formData = new FormData();
    formData.append('file', blob, 'image.jpg');
    const response = await fetch('/predict/', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    emotionLabel.textContent = `${data.emotion} (${(data.percentage * 100).toFixed(2)}%)`;
    requestAnimationFrame(predictEmotion);
}

startButton.addEventListener('click', () => {
    setupCamera();
    predictEmotion();
});

stopButton.addEventListener('click', () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => {
            track.stop();
        });
    }
    emotionLabel.textContent = '';
});
