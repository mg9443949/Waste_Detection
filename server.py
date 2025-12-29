from flask import Flask, Response, request
import requests, time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

# =====================================================
# CONFIG
# =====================================================
ESP32_CAPTURE_URL = "http://10.99.242.240/capture"    # GET image from ESP32
ESP32_NOTIFY_URL  = " https://subglossal-absurdly-janyce.ngrok-free.dev/ai"  # POST result to ESP32

# =====================================================
# MODEL DEFINITION (same as training)
# =====================================================
class WasteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Load model
model = WasteCNN()
model.load_state_dict(torch.load("waste_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head><title>ESP32 Waste Detection</title></head>
<body>
<h2>ESP32 Waste Detection Live</h2>
<img src="/video_feed" width="640" height="480">
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

# =====================================================
# LIVE STREAM GENERATOR
# =====================================================
def generate():
    last_sent = ""

    while True:
        try:
            r = requests.get(ESP32_CAPTURE_URL, timeout=2)
            jpg = r.content
        except Exception as e:
            print("❌ ESP32 capture failed:", e)
            time.sleep(0.5)
            continue

        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # ---- AI INFERENCE ----
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0)

        with torch.no_grad():
            prob = model(tensor).item()

        label = "WASTE" if prob > 0.5 else "NOT WASTE"
        color = (0, 0, 255) if label == "WASTE" else (0, 255, 0)

        # ---- Only send if changed ----
        if label != last_sent:
            try:
                requests.post(ESP32_NOTIFY_URL, json={"result": label}, timeout=0.5)
            except:
                pass
            last_sent = label

        # ---- Draw overlay ----
        cv2.putText(frame, f"{label} ({prob:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        _, jpeg = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
