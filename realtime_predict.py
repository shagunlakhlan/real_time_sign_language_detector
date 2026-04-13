import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter

# ---------------------------
# Load Model
# ---------------------------
with open("model.pkl", "rb") as f:
    model, label_map = pickle.load(f)

# ---------------------------
# MediaPipe Setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2)

# ---------------------------
# Dynamic Buffers
# ---------------------------
buffers = []

# ---------------------------
# Angle Function
# ---------------------------
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:

        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):

            # Ensure buffer exists
            if i >= len(buffers):
                buffers.append(deque(maxlen=7))   # 🔥 slightly bigger buffer

            # Draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ---------------------------
            # Extract Landmarks
            # ---------------------------
            landmarks = []
            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            landmarks = np.array(landmarks)

            # ---------------------------
            # NORMALIZATION
            # ---------------------------
            landmarks = landmarks - landmarks[0]
            norm = np.linalg.norm(landmarks)
            if norm != 0:
                landmarks = landmarks / norm

            points = landmarks.reshape(21, 3)

            # ---------------------------
            # ANGLE FEATURES
            # ---------------------------
            angles = []

            finger_sets = [
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16],
                [17,18,19,20]
            ]

            for finger in finger_sets:
                angles.append(get_angle(points[finger[0]], points[finger[1]], points[finger[2]]))
                angles.append(get_angle(points[finger[1]], points[finger[2]], points[finger[3]]))

            final_features = np.concatenate([landmarks, angles])

            # ---------------------------
            # PREDICTION
            # ---------------------------
            probs = model.predict_proba([final_features])[0]
            confidence = max(probs)
            pred_index = np.argmax(probs)

            predicted_label = label_map[pred_index]

            # ---------------------------
            # UNKNOWN LOGIC
            # ---------------------------
            if confidence < 0.90 or predicted_label == "UNKNOWN":
                buffers[i].append("UNKNOWN")
            else:
                buffers[i].append(predicted_label)

            # ---------------------------
            # SMOOTHING
            # ---------------------------
            text = "Detecting..."

            if len(buffers[i]) == buffers[i].maxlen:
                most_common = Counter(buffers[i]).most_common(1)[0][0]

                text = most_common   # already string (A/B/UNKNOWN)

            # ---------------------------
            # DISPLAY
            # ---------------------------
            cv2.putText(frame, text,
                        (min(x_list), min(y_list) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()