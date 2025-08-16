#!/usr/bin/env python3
"""
Self-Driving Car — Simulator Bridge (Flask-SocketIO version)
- Handles WebSocket connection with Udacity simulator
- Matches preprocessing used in training
- Predicts steering angle + applies throttle
"""

import os
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image

from tensorflow.keras.models import load_model
from flask import Flask
from flask_socketio import SocketIO

# -------------------
# Config
# -------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.h5")
PORT = int(os.environ.get("PORT", 4567))
DEFAULT_THROTTLE = float(os.environ.get("THROTTLE", 0.3))

# -------------------
# App & Model
# -------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
print(f"✅ Loaded model: {MODEL_PATH}")

# -------------------
# Preprocessing
# -------------------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Crop sky/hood
    img = img[60:-25, :, :]
    # Convert to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Resize to Nvidia input
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    # Normalize
    img = img.astype(np.float32) / 255.0
    return img

# -------------------
# Socket.IO handlers
# -------------------
@socketio.on('connect')
def connect():
    print("🚗 Simulator connected")
    socketio.emit("steer", data={
        'steering_angle': '0.0',
        'throttle': str(DEFAULT_THROTTLE)
    })

@socketio.on('telemetry')
def telemetry(data):
    if not data:
        return
    try:
        img_str = data.get("image")
        if img_str is None:
            return

        # Decode image
        image = Image.open(BytesIO(base64.b64decode(img_str)))
        image = image.convert("RGB")
        img_arr = np.asarray(image)

        # Preprocess & predict
        proc = preprocess_image(img_arr)
        proc = np.expand_dims(proc, axis=0)
        steering = float(model.predict(proc, verbose=0)[0][0])

        # Keep throttle constant (can add logic later)
        throttle = DEFAULT_THROTTLE

        print(f"Predicted steering={steering:.4f}, throttle={throttle:.2f}")
        socketio.emit("steer", data={
            'steering_angle': str(steering),
            'throttle': str(throttle)
        })
    except Exception as e:
        print("⚠️ telemetry error:", e)

@socketio.on('disconnect')
def disconnect():
    print("❌ Simulator disconnected")

# -------------------
# Run server
# -------------------
if __name__ == '__main__':
    print(f"🚀 Listening on port {PORT}")
    socketio.run(app, host="0.0.0.0", port=PORT)
