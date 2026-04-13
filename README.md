# 🤟 Real-Time Sign Language Detection System

## 📌 Overview

This project implements a **real-time Sign Language Detection System** that recognizes hand gestures using a webcam and converts them into alphabets.

Unlike traditional image-based models, this system uses a **feature-engineered approach** based on **hand landmark coordinates and joint angles**, making it more efficient and lightweight.

The system is powered by **MediaPipe for hand tracking** and a **Neural Network (MLP Classifier)** for classification, along with a **custom-built GUI** for user interaction.

---

## 🚀 Key Features

- ✋ Real-time hand gesture detection using webcam  
- 🔤 Recognition of A–Z alphabets + UNKNOWN class  
- 🧠 Custom feature extraction (landmarks + angles)  
- ⚡ Lightweight Neural Network (MLPClassifier)  
- 📊 Self-created dataset (CSV format)  
- 💻 GUI-based interaction system  
- 🎯 Fast and efficient prediction  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- MediaPipe  
- NumPy & Pandas  
- Scikit-learn (MLPClassifier)  
- Tkinter (GUI)  

---

## 🧠 Unique Approach (Core Innovation)

Instead of using raw images, this project uses:

### ✔ Hand Landmark Detection
- Extracts **21 key points** of the hand using MediaPipe  

### ✔ Feature Engineering
- Normalized landmark coordinates  
- **Angle calculation between finger joints**  

### ✔ Model Used
- **MLPClassifier (Neural Network)**  
- Hidden layers: `(128, 64)`  
- Trained on engineered features for better accuracy  

This approach:
- Reduces computational cost  
- Improves speed  
- Works well even on low-end systems  

---

## 📊 Dataset

- 📁 **Self-created dataset**
- Stored in CSV format (one file per class)
- Each file contains:
  - Hand landmark coordinates (21 × 3 values)
- Classes include:
  - A – Z alphabets  
  - UNKNOWN  

---

## 📁 Project Structure

```
real_sign_language_project/
│
├── dataset/                  # Self-created dataset (CSV files)
│   ├── A.csv
│   ├── B.csv
│   └── ...
│
├── model.pkl                 # Trained model
│
├── data_collector.py         # Dataset collection
├── train_model.py            # Model training (MLP)
├── realtime_predict.py       # Real-time detection
├── gui.py                    # GUI interface
├── app.py                    # Main app launcher
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/sign-language-detection.git
cd sign-language-detection
pip install mediapipe opencv-python numpy pandas scikit-learn
```

---

## ▶️ Usage

### 1. Data Collection

```bash
python data_collector.py
```

- Collect your own gesture data  
- Saves data in CSV format  

---

### 2. Train Model

```bash
python train_model.py
```

- Extracts features (coordinates + angles)  
- Trains MLP model  
- Saves model as `model.pkl`  

---

### 3. Real-Time Prediction

```bash
python realtime_predict.py
```

- Opens webcam  
- Detects hand landmarks  
- Predicts gesture in real time  

---

### 4. Run GUI

```bash
python app.py
```

- Launches user-friendly interface  

---

## 🎯 Objectives

- Build an efficient sign language recognition system  
- Use feature engineering instead of heavy deep learning  
- Enable real-time communication assistance  
- Provide a lightweight and scalable solution  

---

## 🔮 Future Scope

- Word and sentence formation  
- Speech output (Text-to-Speech)  
- Deep learning hybrid models  
- Mobile application version  
- Multi-hand detection  

---

## 👨‍💻 Author

**Shagun**  
B.Tech Computer Science (Artificial Intelligence)
