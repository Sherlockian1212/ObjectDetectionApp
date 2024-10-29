import base64
import io
import os
from flask import Flask, request, jsonify, render_template
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

    detected_objects = []
    class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    # Phát hiện vật thể
    results = model(image)
    for result in results:
        if result:
            class_ids = result.boxes.cls
            for class_id in class_ids:
                class_name = class_names[int(class_id)]
                detected_objects.append(class_name)
            
    # Đếm số lượng từng đối tượng
    object_counts = {}
    for obj in detected_objects:
        if obj in object_counts:
            object_counts[obj] += 1
        else:
            object_counts[obj] = 1

    # Chuyển thành chuỗi văn bản dạng "two person, two chair"
    detected_summary = ', '.join([f"{count} {name}" for name, count in object_counts.items()])
    print(detected_summary)
    if detected_summary:
        speak_results(detected_summary)

    return jsonify({"detected_objects": detected_summary})

def speak_results(objects):
    tts = gTTS(objects, lang='en')
    if os.path.exists("result.mp3"):
        os.remove("result.mp3")
    tts.save("result.mp3")
    playsound("result.mp3")

if __name__ == "__main__":
    app.run(debug=True)
