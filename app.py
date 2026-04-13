import streamlit as st
import cv2
import mediapipe as mp       # type: ignore
import numpy as np
import pickle
from collections import deque, Counter

# ---------------------------
# Load Model
# ---------------------------
with open("model.pkl", "rb") as f:
    model, label_map = pickle.load(f)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🤟 Sign Language Detection System")
st.markdown("Real-time hand sign recognition using webcam")

run = st.button("Start Camera")

FRAME_WINDOW = st.image([])

# ---------------------------
# MediaPipe
# ---------------------------
mp_hands = mp.solutions.hands   # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore

hands = mp_hands.Hands(max_num_hands=2)

buffers = []

# ---------------------------
# Webcam
# ---------------------------
if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:

            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):

                # Create buffer if needed
                if i >= len(buffers):
                    buffers.append(deque(maxlen=5))

                # Draw skeleton
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                landmarks = []
                x_list, y_list = [], []

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                landmarks = np.array(landmarks)
                landmarks = landmarks - landmarks[0]

                # Prediction
                probs = model.predict_proba([landmarks])[0]
                confidence = max(probs)
                pred_index = np.argmax(probs)

                if confidence > 0.75:
                    buffers[i].append(pred_index)
                else:
                    buffers[i].append(-1)

                text = "Detecting..."

                if len(buffers[i]) == buffers[i].maxlen:
                    most_common = Counter(buffers[i]).most_common(1)[0][0]

                    if most_common != -1:
                        label_pred = label_map[most_common]
                        text = f"{label_pred} ({confidence:.2f})"
                    else:
                        text = "Unknown"

                # Show prediction
                cv2.putText(frame, text,
                            (min(x_list), min(y_list) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

        # Convert BGR → RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()