import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# ---------------------------
# MediaPipe Setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

# ---------------------------
# Dataset Setup
# ---------------------------
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# 🔥 A-Z + UNKNOWN
labels = [chr(i) for i in range(65, 91)] + ["UNKNOWN"]

current_label_index = 0
samples_collected = 0
target_samples = 300

recording = False

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

print("\nControls:")
print("S → Start/Stop Recording")
print("N → Next Label")
print("Q → Quit\n")

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    label = labels[current_label_index]

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ---------------------------
            # Extract Landmarks
            # ---------------------------
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # ---------------------------
            # NORMALIZATION
            # ---------------------------
            landmarks = landmarks - landmarks[0]
            norm = np.linalg.norm(landmarks)
            if norm != 0:
                landmarks = landmarks / norm

            # ---------------------------
            # SAVE DATA
            # ---------------------------
            if recording and samples_collected < target_samples:
                file_path = os.path.join(dataset_path, f"{label}.csv")

                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)

                samples_collected += 1

    # ---------------------------
    # DISPLAY INFO
    # ---------------------------
    status = "Recording" if recording else "Paused"

    cv2.putText(frame, f"Label: {label}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Samples: {samples_collected}/{target_samples}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"Status: {status}", (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Dataset Collector", frame)

    # ---------------------------
    # CONTROLS
    # ---------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        current_label_index = (current_label_index + 1) % len(labels)
        samples_collected = 0
        recording = False
        print(f"\n➡ Switched to: {labels[current_label_index]}")

    elif key == ord('s'):
        recording = not recording
        print("Recording:", recording)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()