"""
gui.py — Sign Language Detection Suite
======================================
Drop this file into your real_sign_language_project/ folder.
Run with:  python gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pickle
import csv
import os
import subprocess
import sys
from collections import deque, Counter
from PIL import Image, ImageTk
from typing import Optional
from sklearn.neural_network import MLPClassifier


# ─────────────────────────────────────────────
# THEME  (dark, electric-teal accent)
# ─────────────────────────────────────────────
BG        = "#0d0f14"
SURFACE   = "#161b24"
SURFACE2  = "#1e2535"
ACCENT    = "#00e5cc"
ACCENT2   = "#0099aa"
TEXT      = "#e8edf5"
SUBTEXT   = "#7a8499"
RED       = "#ff4757"
GREEN     = "#2ed573"
YELLOW    = "#ffa502"
FONT_HEAD = ("Courier New", 22, "bold")
FONT_SUB  = ("Courier New", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_BIG  = ("Courier New", 36, "bold")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def extract_features(row):
    row = np.array(row)
    row = row - row[0]
    norm = np.linalg.norm(row)
    if norm != 0:
        row = row / norm
    points = row.reshape(21, 3)
    angles = []
    for finger in [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]:
        angles.append(get_angle(points[finger[0]], points[finger[1]], points[finger[2]]))
        angles.append(get_angle(points[finger[1]], points[finger[2]], points[finger[3]]))
    return np.concatenate([row, angles])


def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────

class SignLangApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language Detection Suite")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.resizable(True, True)

        # shared state
        self.model, self.label_map = load_model()  # type: ignore
        self._camera_running = False
        self._collector_running = False
        self._cap = None
        self._thread = None

        self._build_ui()

    # ── BUILD ──────────────────────────────────

    def _build_ui(self):
        # ── LEFT SIDEBAR ──
        sidebar = tk.Frame(self, bg=SURFACE, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🤟", font=("Segoe UI Emoji", 36),
                 bg=SURFACE, fg=ACCENT).pack(pady=(28, 4))
        tk.Label(sidebar, text="SIGN LANG\nSUITE", font=("Courier New", 13, "bold"),
                 bg=SURFACE, fg=ACCENT, justify="center").pack()
        tk.Label(sidebar, text="v2.0", font=FONT_SUB, bg=SURFACE,
                 fg=SUBTEXT).pack(pady=(0, 30))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=20)

        self._nav_btns = {}
        nav_items = [
            ("🎯  Detect",       "detect"),
            ("ℹ️   About",        "about"),
        ]
        self._active_tab = tk.StringVar(value="detect")

        for label, tab_id in nav_items:
            btn = tk.Button(
                sidebar, text=label, font=("Segoe UI", 11),
                bg=SURFACE, fg=TEXT, activebackground=SURFACE2,
                activeforeground=ACCENT, relief="flat", anchor="w",
                padx=20, pady=12, cursor="hand2",
                command=lambda t=tab_id: self._switch_tab(t)
            )
            btn.pack(fill="x", pady=1)
            self._nav_btns[tab_id] = btn

        # status dot
        self._status_frame = tk.Frame(sidebar, bg=SURFACE)
        self._status_frame.pack(side="bottom", pady=20, fill="x", padx=20)
        tk.Label(self._status_frame, text="Model", font=FONT_SUB,
                 bg=SURFACE, fg=SUBTEXT).pack(anchor="w")
        self._model_status = tk.Label(
            self._status_frame,
            text="● Loaded" if self.model else "● Not found",
            font=("Segoe UI", 10, "bold"),
            bg=SURFACE,
            fg=GREEN if self.model else RED
        )
        self._model_status.pack(anchor="w")

        # ── RIGHT CONTENT ──
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(side="right", fill="both", expand=True)

        # pages dict
        self._pages = {}
        for tab_id in ["detect", "collect", "train", "about"]:
            page = tk.Frame(self._content, bg=BG)
            page.place(relwidth=1, relheight=1)
            self._pages[tab_id] = page

        self._build_detect_page()
        self._build_about_page()

        self._switch_tab("detect")

    # ── NAV ───────────────────────────────────

    def _switch_tab(self, tab_id):
        # stop camera if leaving detect/collect
        if tab_id not in ("detect", "collect"):
            self._stop_camera()

        for tid, btn in self._nav_btns.items():
            if tid == tab_id:
                btn.configure(bg=SURFACE2, fg=ACCENT)
            else:
                btn.configure(bg=SURFACE, fg=TEXT)

        for tid, page in self._pages.items():
            if tid == tab_id:
                page.lift()
            else:
                page.lower()

        self._active_tab.set(tab_id)

    # ══════════════════════════════════════════
    # PAGE: DETECT
    # ══════════════════════════════════════════

    def _build_detect_page(self):
        p = self._pages["detect"]

        # header
        hdr = tk.Frame(p, bg=BG)
        hdr.pack(fill="x", padx=30, pady=(25, 10))
        tk.Label(hdr, text="Real-Time Detection", font=FONT_HEAD,
                 bg=BG, fg=TEXT).pack(side="left")

        # body
        body = tk.Frame(p, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=10)

        # camera frame
        cam_container = tk.Frame(body, bg=SURFACE2, bd=0, width=640, height=480)
        cam_container.pack(side="left", fill="both", expand=True)
        cam_container.pack_propagate(False)

        self._detect_canvas = tk.Label(cam_container, bg=SURFACE2,
                                       text="Camera feed will appear here",
                                       font=FONT_BODY, fg=SUBTEXT)
        self._detect_canvas.pack(fill="both", expand=True)

        # right panel
        rp = tk.Frame(body, bg=BG, width=240)
        rp.pack(side="right", fill="y", padx=(16, 0))
        rp.pack_propagate(False)

        # prediction card
        pred_card = tk.Frame(rp, bg=SURFACE, bd=0, highlightbackground=ACCENT,
                             highlightthickness=1)
        pred_card.pack(fill="x", pady=(0, 12))

        tk.Label(pred_card, text="PREDICTION", font=FONT_SUB,
                 bg=SURFACE, fg=SUBTEXT).pack(anchor="w", padx=14, pady=(12,0))

        self._pred_label = tk.Label(pred_card, text="—", font=FONT_BIG,
                                    bg=SURFACE, fg=ACCENT)
        self._pred_label.pack(padx=14, pady=6)

        self._conf_label = tk.Label(pred_card, text="confidence: —",
                                    font=FONT_SUB, bg=SURFACE, fg=SUBTEXT)
        self._conf_label.pack(anchor="w", padx=14, pady=(0, 14))

        # history card
        hist_card = tk.Frame(rp, bg=SURFACE)
        hist_card.pack(fill="x", pady=(0, 12))
        tk.Label(hist_card, text="HISTORY", font=FONT_SUB,
                 bg=SURFACE, fg=SUBTEXT).pack(anchor="w", padx=14, pady=(10,4))
        self._history_text = tk.Label(hist_card, text="", font=("Courier New", 14, "bold"),
                                      bg=SURFACE, fg=TEXT, wraplength=200, justify="left")
        self._history_text.pack(anchor="w", padx=14, pady=(0, 10))

        self._history_buffer = []

        btn_row = tk.Frame(rp, bg=BG)
        btn_row.pack(fill="x")

        self._detect_btn = tk.Button(
            btn_row, text="▶  START", font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg=BG, activebackground=ACCENT2, activeforeground=BG,
            relief="flat", padx=12, pady=10, cursor="hand2",
            command=self._toggle_detect
        )
        self._detect_btn.pack(fill="x", pady=3)

        tk.Button(
            btn_row, text="■  STOP CAMERA", font=("Segoe UI", 10, "bold"),
            bg=RED, fg="white", activebackground="#cc3344",
            relief="flat", padx=12, pady=9, cursor="hand2",
            command=self._stop_and_reset
        ).pack(fill="x", pady=3)


        tk.Button(
            btn_row, text="🗑  Clear History", font=FONT_BODY,
            bg=SURFACE, fg=TEXT, activebackground=SURFACE2, activeforeground=ACCENT,
            relief="flat", padx=12, pady=8, cursor="hand2",
            command=self._clear_history
        ).pack(fill="x", pady=3)

    def _toggle_detect(self):
        if self._camera_running:
            self._stop_camera()
            self._detect_btn.configure(text="▶  START", bg=ACCENT, fg=BG)
        else:
            if not self.model:
                messagebox.showerror("Model Missing",
                    "model.pkl not found.\nPlease train the model first.")
                return
            self._detect_btn.configure(text="■  STOP", bg=RED, fg="white")
            self._start_detect_camera()

    def _start_detect_camera(self):
        self._camera_running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def _detect_loop(self):
        mp_hands = mp.solutions.hands  # type: ignore
        mp_draw  = mp.solutions.drawing_utils # type: ignore
        hands    = mp_hands.Hands(max_num_hands=2,
                                  min_detection_confidence=0.6,
                                  min_tracking_confidence=0.6)
        buffers  = []

        cap = cv2.VideoCapture(0)
        self._cap = cap

        while self._camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            current_pred = "—"
            current_conf = 0.0

            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    if i >= len(buffers):
                        buffers.append(deque(maxlen=7))

                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,229,204), thickness=2),
                        mp_draw.DrawingSpec(color=(255,255,255), thickness=1)
                    )

                    lms = []
                    x_list, y_list = [], []
                    for lm in hand_landmarks.landmark:
                        lms.extend([lm.x, lm.y, lm.z])
                        x_list.append(int(lm.x * w))
                        y_list.append(int(lm.y * h))

                    feats = extract_features(lms)
                    probs = self.model.predict_proba([feats])[0] # type: ignore
                    confidence = max(probs)
                    pred_index = np.argmax(probs)
                    predicted  = self.label_map[pred_index] # type: ignore

                    if confidence < 0.90 or predicted == "UNKNOWN":
                        buffers[i].append("?")
                    else:
                        buffers[i].append(predicted)

                    text = "..."
                    if len(buffers[i]) == buffers[i].maxlen:
                        most_common = Counter(buffers[i]).most_common(1)[0][0]
                        text = most_common

                    current_pred = text
                    current_conf = confidence

                    # draw box + label on frame
                    x1 = max(min(x_list) - 20, 0)
                    y1 = max(min(y_list) - 20, 0)
                    x2 = min(max(x_list) + 20, w)
                    y2 = min(max(y_list) + 20, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 204), 2)

                    # overlay bar
                    bar_h = 36
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - bar_h), (x2, y1), (13, 15, 20), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    cv2.putText(frame, text,
                                (x1 + 6, y1 - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (0, 229, 204), 2)

            # convert to tk image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            canvas_w = self._detect_canvas.winfo_width()
            canvas_h = self._detect_canvas.winfo_height()
            if canvas_w > 10 and canvas_h > 10:
                img = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)

            if self._camera_running:
                self._detect_canvas.configure(image=imgtk)
                self._detect_canvas.image = imgtk  # type: ignore
                self._update_pred_ui(current_pred, current_conf)

        cap.release()
        self._cap = None
        self._camera_running = False

    def _update_pred_ui(self, pred, conf):
        def _do():
            self._pred_label.configure(text=pred if pred else "—")
            self._conf_label.configure(text=f"confidence: {conf:.2%}" if conf else "confidence: —")

            if pred and pred not in ("—", "...", "?") and (
                len(self._history_buffer) == 0 or
                (self._history_buffer and pred != self._history_buffer[-1])
            ):
                self._history_buffer.append(pred)
                if len(self._history_buffer) > 20:
                    self._history_buffer.pop(0)
                self._history_text.configure(text=" ".join(self._history_buffer[-12:]))

        self.after(0, _do)

    def _clear_history(self):
        self._history_buffer = []
        self._history_text.configure(text="")

    def _stop_and_reset(self):
        self._stop_camera()
        self._detect_btn.configure(text="▶  START", bg=ACCENT, fg=BG)
        self._detect_canvas.configure(image="", text="Camera feed will appear here",
                                      font=FONT_BODY, fg=SUBTEXT)


    # ══════════════════════════════════════════
    # PAGE: ABOUT
    # ══════════════════════════════════════════

    def _build_about_page(self):
        p = self._pages["about"]

        # scrollable container
        canvas = tk.Canvas(p, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(p, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        outer = tk.Frame(canvas, bg=BG)
        canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_resize)
        outer.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        def section_title(text):
            tk.Label(outer, text=text, font=("Courier New", 13, "bold"),
                     bg=BG, fg=ACCENT).pack(anchor="w", padx=40, pady=(28, 6))
            tk.Frame(outer, bg=ACCENT, height=1).pack(fill="x", padx=40, pady=(0, 10))

        def body_text(text):
            tk.Label(outer, text=text, font=("Segoe UI", 10),
                     bg=BG, fg=TEXT, justify="left", wraplength=700).pack(
                anchor="w", padx=40, pady=3)

        def step_row(number, title, desc):
            row = tk.Frame(outer, bg=SURFACE, highlightbackground=SURFACE2,
                           highlightthickness=1)
            row.pack(fill="x", padx=40, pady=4)
            tk.Label(row, text=str(number), font=("Courier New", 20, "bold"),
                     bg=SURFACE, fg=ACCENT, width=3).pack(side="left", padx=(12,0), pady=12)
            txt = tk.Frame(row, bg=SURFACE)
            txt.pack(side="left", padx=12, pady=10)
            tk.Label(txt, text=title, font=("Segoe UI", 11, "bold"),
                     bg=SURFACE, fg=TEXT).pack(anchor="w")
            tk.Label(txt, text=desc, font=("Segoe UI", 10),
                     bg=SURFACE, fg=SUBTEXT, wraplength=560, justify="left").pack(anchor="w")

        def info_pill(icon, label, value):
            row = tk.Frame(outer, bg=SURFACE2)
            row.pack(side="left", padx=(40 if not outer.winfo_children() else 6), pady=4)
            tk.Label(row, text=f"{icon}  {label}", font=FONT_SUB,
                     bg=SURFACE2, fg=SUBTEXT).pack(anchor="w", padx=10, pady=(8,2))
            tk.Label(row, text=value, font=("Segoe UI", 10, "bold"),
                     bg=SURFACE2, fg=TEXT).pack(anchor="w", padx=10, pady=(0,8))

        # ── HERO ──
        tk.Label(outer, text="🤟", font=("Segoe UI Emoji", 52),
                 bg=BG).pack(pady=(36, 4))
        tk.Label(outer, text="Sign Language Detection Suite",
                 font=("Courier New", 22, "bold"), bg=BG, fg=TEXT).pack()
        tk.Label(outer, text="Real-time ASL alphabet recognition powered by MediaPipe & Machine Learning",
                 font=("Segoe UI", 10), bg=BG, fg=SUBTEXT).pack(pady=(4, 0))

        # ── ABOUT ──
        section_title("About")
        body_text(
            "This application detects American Sign Language (ASL) hand signs in real time "
            "using your webcam. It tracks 21 hand landmarks per hand using Google's MediaPipe "
            "framework, extracts finger-joint angles and normalized coordinates as features, "
            "and classifies them using a trained Multi-Layer Perceptron (MLP) neural network."
        )
        body_text(
            "The model supports all 26 letters of the alphabet (A–Z) plus an UNKNOWN class "
            "for signs it cannot confidently identify. A smoothing buffer ensures predictions "
            "are stable and not jumpy frame-to-frame."
        )

        # ── HOW TO USE ──
        section_title("How to Use")


        step_row(1, "Go to the Detect tab",
                 "Click 'Detect' in the left sidebar to open the real-time detection screen.")

        step_row(2, "Press ▶ START",
                 "Your webcam will activate. Hold your hand clearly in front of the camera "
                 "with good lighting. The app works best with a plain background.")

        step_row(3, "Show a sign",
                 "Hold a static ASL letter sign steady for a moment. The prediction panel "
                 "on the right will display the detected letter and confidence score.")

        step_row(4, "Read the History strip",
                 "Every new unique letter detected gets appended to the History panel, "
                 "letting you spell out words sign by sign.")

        step_row(5, "Press ■ STOP when done",
                 "Click the button again to turn off the webcam.")

        # ── TIPS ──
        section_title("Tips for Best Accuracy")
        tips = [
            "💡  Use good, even lighting — avoid strong backlighting.",
            "💡  Keep your hand fully visible and centred in the frame.",
            "💡  Hold each sign steady for at least half a second.",
            "💡  A plain or neutral background improves detection.",
            "💡  Confidence threshold is set to 90% — signs below this show as UNKNOWN.",
        ]
        for tip in tips:
            tk.Label(outer, text=tip, font=("Segoe UI", 10),
                     bg=BG, fg=SUBTEXT).pack(anchor="w", padx=48, pady=2)

        # ── TECH STACK ──
        section_title("Tech Stack")
        stack = [
            ("🖐", "MediaPipe",    "Hand landmark detection"),
            ("🤖", "scikit-learn", "MLP classifier"),
            ("📷", "OpenCV",       "Webcam capture & drawing"),
            ("🐍", "Python",       "Core language"),
            ("🪟", "Tkinter",      "GUI framework"),
        ]
        pills_row = tk.Frame(outer, bg=BG)
        pills_row.pack(fill="x", padx=40, pady=6)
        for icon, name, desc in stack:
            card = tk.Frame(pills_row, bg=SURFACE2)
            card.pack(side="left", padx=(0, 8))
            tk.Label(card, text=icon, font=("Segoe UI Emoji", 18),
                     bg=SURFACE2).pack(pady=(10,2), padx=16)
            tk.Label(card, text=name, font=("Segoe UI", 10, "bold"),
                     bg=SURFACE2, fg=TEXT).pack(padx=16)
            tk.Label(card, text=desc, font=FONT_SUB,
                     bg=SURFACE2, fg=SUBTEXT).pack(padx=16, pady=(0,10))

        # bottom padding
        tk.Frame(outer, bg=BG, height=40).pack()
    # ── SHARED CAMERA STOP ─────────────────────

    def _stop_camera(self):
        self._camera_running = False
        self._recording = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def on_close(self):
        self._stop_camera()
        self.destroy()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk

    app = SignLangApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()