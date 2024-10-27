const video = document.getElementById('camera');
const canvas = document.getElementById('snapshot');
const context = canvas.getContext('2d');

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
        video.addEventListener('click', capturePhoto);
    } catch (error) {
        console.error("Error accessing camera:", error);
    }
}

function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');
    sendImageToServer(imageData);
}

async function sendImageToServer(imageData) {
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();
        console.log("Detected objects:", result.detected_objects);
    } catch (error) {
        console.error("Error sending image to server:", error);
    }
}

startCamera();
