# 🤟 Sign Language Detection System (ASL)

## 📌 Overview

This project is a **real-time American Sign Language (ASL) detection system** using **Computer Vision and Machine Learning**.

It detects hand gestures from webcam input and converts them into **alphabet characters or words**.

---

## 🚀 Features

* Real-time hand tracking using MediaPipe
* ASL alphabet recognition
* Dataset collection tool
* Model training pipeline
* Live prediction system
* Word formation from detected letters

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* TensorFlow Lite
* NumPy

---

## 📂 Project Structure

```
NEW/
│
├── collect_data.py        # Collect ASL dataset
├── train_model.py         # Train ML model
├── main2.py               # Real-time prediction (Random Forest)
├── translator.py          # Advanced TFLite translator
├── asl_model.pkl          # Trained model
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

### 1️⃣ Collect Data

Run:

```
python collect_data.py
```

* Press a-z keys to record letters
* Saves data in CSV format

---

### 2️⃣ Train Model

```
python train_model.py
```

* Trains Random Forest model
* Saves model as `.pkl`

---

### 3️⃣ Run Detection

```
python main2.py
```

👉 Uses trained model for real-time prediction

---

### 4️⃣ Advanced Translator

```
python translator.py
```

👉 Uses TensorFlow Lite for better performance

---

## 📦 Requirements

Install dependencies:

```
pip install -r requirements.txt
```

---

## ⚠️ Notes

* Webcam required
* Ensure model files exist:

  * `asl_model.pkl`
  * `keypoint_classifier.tflite` (for advanced mode)
* Proper lighting improves accuracy

---

## 🔮 Future Improvements

* Sentence generation using NLP
* Mobile app integration
* Multi-language sign detection
* Deployment on edge devices

---

## 👨‍💻 Author

**Tathagata Mandal**

---

## ⭐ Description

A complete pipeline from **data collection → model training → real-time inference**, designed for accessibility and AI-based communication.

