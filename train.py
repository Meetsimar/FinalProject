#!/usr/bin/env python3
"""
Self-Driving Car — Training Script (refactored)
- Robust CSV/path handling for Udacity simulator
- Balanced sampling + augmentation
- Nvidia architecture (200x66 YUV)
- Clear logging + plots + checkpoints

Tested on macOS with TensorFlow (CPU/MPS). Requires: tensorflow>=2.12, opencv-python, pillow, scikit-learn, matplotlib, numpy, python-socketio (for testing script), flask.
"""
import os
import csv
import math
import random
import glob
from typing import List, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# -----------------
# Config
# -----------------
DATA_DIR = os.environ.get("DATA_DIR", "data")  # folder containing IMG/ and driving_log.csv
CSV_NAME = os.environ.get("CSV_NAME", "driving_log.csv")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("EPOCHS", 12))
LR = float(os.environ.get("LR", 1e-4))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.2))
STEERING_CORRECTION = float(os.environ.get("STEER_CORR", 0.2))  # for left/right camera
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------
# Utilities
# -----------------

def _resolve_img_path(raw_path: str) -> str:
    """Resolve image path from CSV row to actual file in DATA_DIR/IMG.
    Handles absolute paths, different separators, and trims spaces.
    """
    p = raw_path.strip().replace("\\", "/")
    base = os.path.basename(p)
    candidate = os.path.join(DATA_DIR, "IMG", base)
    if os.path.exists(candidate):
        return candidate
    # Fallback: search recursively (handles custom data layouts)
    matches = glob.glob(os.path.join(DATA_DIR, "**", base), recursive=True)
    if matches:
        return matches[0]
    return p  # last resort; may be absolute path already


def load_samples(csv_path: str) -> List[Tuple[str, str, str, float]]:
    """Load (center,left,right,steering) tuples. Skips header if present."""
    samples = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            # Skip header row if it contains text
            if row[0].lower().endswith('.jpg') is False and row[3].replace('.', '', 1).lstrip('-').isdigit() is False:
                continue
            center, left, right = row[0], row[1], row[2]
            steering = float(row[3])
            samples.append((center, left, right, steering))
    if not samples:
        raise FileNotFoundError(f"No valid samples found in {csv_path}")
    return samples

# -----------------
# Preprocessing & Augmentation
# -----------------

def crop_resize_normalize(img: np.ndarray) -> np.ndarray:
    # Expect RGB input
    img = img[60:-25, :, :]  # crop sky/hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def random_flip(image: np.ndarray, angle: float):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def random_brightness(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_shift(image: np.ndarray, angle: float, max_shift: int = 20):
    # Horizontal/vertical shift to simulate camera panning
    tx = np.random.randint(-max_shift, max_shift + 1)
    ty = np.random.randint(-max_shift // 2, max_shift // 2 + 1)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    angle += tx * 0.002  # small steering adjustment per pixel shift
    return shifted, angle


def augment(image: np.ndarray, angle: float):
    image, angle = random_flip(image, angle)
    if np.random.rand() < 0.9:
        image = random_brightness(image)
    if np.random.rand() < 0.9:
        image, angle = random_shift(image, angle)
    return image, angle

# -----------------
# Generator
# -----------------

def generator(samples, batch_size=BATCH_SIZE, training=True):
    num = len(samples)
    while True:
        random.shuffle(samples)
        for offset in range(0, num, batch_size):
            batch = samples[offset:offset + batch_size]
            images, angles = [], []
            for center, left, right, steering in batch:
                # Randomly choose camera for robustness
                cam_choice = np.random.choice(['center', 'left', 'right']) if training else 'center'
                if cam_choice == 'center':
                    img_path = _resolve_img_path(center)
                    angle = steering
                elif cam_choice == 'left':
                    img_path = _resolve_img_path(left)
                    angle = steering + STEERING_CORRECTION
                else:
                    img_path = _resolve_img_path(right)
                    angle = steering - STEERING_CORRECTION

                img = cv2.imread(img_path)
                if img is None:
                    # try alternate resolution or path; skip if unreadable
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if training:
                    # Apply augmentation on pre-crop RGB to keep color space consistent
                    img_aug, angle_aug = augment(img.copy(), angle)
                    img_proc = crop_resize_normalize(img_aug)
                    images.append(img_proc)
                    angles.append(angle_aug)
                else:
                    img_proc = crop_resize_normalize(img)
                    images.append(img_proc)
                    angles.append(angle)

            if not images:
                # Avoid empty batch
                continue
            yield np.array(images), np.array(angles, dtype=np.float32)

# -----------------
# Model (Nvidia Architecture)
# -----------------

def nvidia_model():
    model = Sequential([
        Lambda(lambda x: x, input_shape=(66, 200, 3)),
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    return model

# -----------------
# Steering Histogram (diagnostic)
# -----------------

def plot_steering_histogram(samples):
    angles = [s[3] for s in samples]
    plt.figure()
    plt.hist(angles, bins=31)
    plt.title("Steering Angle Distribution")
    plt.xlabel("Angle")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("steering_histogram.png")
    print("Saved steering_histogram.png")

# -----------------
# Main
# -----------------

def main():
    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    samples = load_samples(csv_path)
    print(f"Loaded {len(samples)} samples from {csv_path}")

    plot_steering_histogram(samples)

    train_samples, val_samples = train_test_split(samples, test_size=VAL_SPLIT, random_state=SEED, shuffle=True)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_gen = generator(train_samples, training=True)
    val_gen = generator(val_samples, training=False)

    model = nvidia_model()
    model.compile(loss='mse', optimizer=Adam(learning_rate=LR))

    steps_per_epoch = max(1, len(train_samples) // BATCH_SIZE)
    val_steps = max(1, len(val_samples) // BATCH_SIZE)

    ckpt = ModelCheckpoint(
        filepath="best_model.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=[ckpt, early, rlrop],
        verbose=1
    )

    # Save final model
    model.save("self_driving_model.h5")
    print("Saved self_driving_model.h5 and best_model.h5 (best on val)")

    # Plot training curves
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")


if __name__ == '__main__':
    main()