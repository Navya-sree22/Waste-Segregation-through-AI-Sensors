import cv2
import numpy as np
import os
import time
import serial
from tensorflow.keras.models import load_model

# ===============================
# 1. Arduino Setup
# ===============================
# Replace 'COM3' with the correct port (e.g., 'COM4', '/dev/ttyUSB0' for Linux)
#arduino = serial.Serial('COM3', 9600, timeout=1)
#time.sleep(2)  # Wait for Arduino to initialize

# ===============================
# 2. Model Setup
# ===============================
model = load_model(r"C:\Users\hp\OneDrive\Desktop\AIML\best_model.keras")
class_names = ['Non-Recyclable', 'Organic', 'Plastic', 'Recyclable']
IMG_SIZE = 224

# ===============================
# 3. Output Folder Setup
# ===============================
os.makedirs("captures", exist_ok=True)
capture_count = 1

# ===============================
# 4. Debounce & Webcam Setup
# ===============================
last_confident_detection_time = {}
COOLDOWN_PERIOD = 5  # seconds
current_display_label = "Unknown"
current_display_conf = 0.0
last_displayed_update_time = 1.0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Waste Detector Started!")
print("Press 'q' to quit | 's' to save frame")
print(f"Debounce time: {COOLDOWN_PERIOD}s\n")

# ===============================
# 5. Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting...")
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    left = max(0, cx - IMG_SIZE // 2)
    top = max(0, cy - IMG_SIZE // 2)
    right = min(w, cx + IMG_SIZE // 2)
    bottom = min(h, cy + IMG_SIZE // 2)

    cropped = frame[top:bottom, left:right]
    if cropped.shape[0] != IMG_SIZE or cropped.shape[1] != IMG_SIZE:
        cv2.putText(frame, "Adjust camera (Crop too small)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('♻️ Waste Detector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("⚠️ Cannot save: Invalid crop area.")
        continue

    # Preprocess and predict
    img = cropped / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    conf = np.max(pred)
    label_id = np.argmax(pred)
    label = class_names[label_id]

    current_time = time.time()

    if conf > 0.4:
        if label not in last_confident_detection_time or \
           (current_time - last_confident_detection_time[label]) > COOLDOWN_PERIOD:
            print(f"Detected: {label} ({conf * 100:.1f}%)")
            last_confident_detection_time[label] = current_time

            # ✅ Send command to Arduino
            #if label == "Plastic":
             #   arduino.write(b'1\n')
            #elif label == "Organic":
             #   arduino.write(b'2\n')
            #elif label == "Recyclable":
             #   arduino.write(b'3\n')
            #elif label == "Non-Recyclable":
             #   arduino.write(b'4\n')

            current_display_label = label
            current_display_conf = conf
            last_displayed_update_time = current_time
    else:
        if current_display_label != "Unknown" and \
           (current_time - last_displayed_update_time) > COOLDOWN_PERIOD:
            current_display_label = "Unknown"
            current_display_conf = 0.0

    # Draw bounding box and label
    display_text = f"{current_display_label} ({current_display_conf * 100:.1f}%)"
    color = (0, 255, 0) if current_display_label != "Unknown" else (0, 0, 255)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    cv2.imshow('♻️ Waste Detector', frame)

    # Key Events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"captures/{current_display_label}_{capture_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")
        capture_count += 1

# ===============================
# 6. Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
arduino.close()
print("👋 Closed all connections and exited.")
