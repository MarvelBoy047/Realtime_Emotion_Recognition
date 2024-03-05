var imgElement = document.getElementById('videoElement');
var emotionElement = document.getElementById('emotion');
var startButton = document.getElementById('startButton');
var stopButton = document.getElementById('stopButton');

// Function to update the webcam feed URL
function updateVideoSource(url) {
    imgElement.src = url;
}

startButton.onclick = function() {
    fetch('/start_detection', {
        method: 'POST'
    }).then(function(response) {
        if (response.ok) {
            // If the request is successful, update the webcam feed URL
            updateVideoSource("{{ url_for('video_feed') }}");
        }
    });
};

stopButton.onclick = function() {
    fetch('/stop_detection', {
        method: 'POST'
    }).then(function(response) {
        if (response.ok) {
            // If the request is successful, clear the webcam feed URL
            updateVideoSource("");
        }
    });
};

// Periodically update the detected emotion
setInterval(function() {
    fetch('/detected_emotion')
        .then(response => response.json())
        .then(data => {
            emotionElement.innerText = 'Detected Emotion: ' + data.emotion;
        })
        .catch(error => console.error('Error:', error));
}, 1000); // Adjust the interval as needed
