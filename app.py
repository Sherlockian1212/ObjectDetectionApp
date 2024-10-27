import base64
import io
import os
import torch
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
from gtts import gTTS
from ultralytics import YOLO
from playsound import playsound

app = Flask(__name__)

# Đường dẫn đến file mô hình YOLOv5 local của bạn
model_path = 'yolo11n.pt'  # Ví dụ: 'models/yolov5s.pt'

# Tải mô hình từ file local
model = YOLO("yolo11n.pt")

@app.route('/')
def index():
    return render_template('index.html')  # Trả về index.html

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Phát hiện vật thể
    results = model(image)
    detected_objects = ["shirt"]
    
    if detected_objects:
        speak_results(detected_objects)

    return jsonify({"detected_objects": detected_objects})

def speak_results(objects):
    text = ', '.join(objects)
    tts = gTTS(text, lang='en')
    tts.save("result.mp3")
    playsound("result.mp3")

if __name__ == "__main__":
    app.run(debug=True)
