# 🤟 Sign Language Detection System (ASL)

## 📌 Overview

This project is a **real-time American Sign Language (ASL) recognition system** built using **Computer Vision and Machine Learning**.

It captures hand gestures from a webcam, processes them using **MediaPipe hand tracking**, and predicts the corresponding alphabet using trained models.

The system provides a **complete pipeline**:
➡️ Data Collection → Model Training → Real-Time Prediction → Advanced Translation

---

## 🎯 Objective

To build an **AI-based assistive system** that helps bridge the communication gap between **hearing-impaired and non-sign-language users**, improving accessibility and inclusivity.

---

## 🚀 Features

* 📷 Real-time hand tracking using MediaPipe
* 🔤 ASL alphabet recognition (A–Z)
* 🧠 Machine Learning model (Random Forest)
* ⚡ TensorFlow Lite optimized translator
* 📝 Dataset collection tool
* 🔄 End-to-end ML pipeline
* 📡 Live prediction from webcam

---

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe**
* **Scikit-learn**
* **TensorFlow Lite**
* **NumPy**

---

## 📂 Project Structure

```
Sign_Language_Detection_System/
│
├── collect_data.py        # Dataset collection using webcam
├── train_model.py         # Train ML model (Random Forest)
├── main2.py               # Real-time prediction (using .pkl model)
├── translator.py          # Advanced TFLite-based translator
├── asl_model.pkl          # Trained ML model
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1️⃣ **Hand Detection**
MediaPipe detects 21 hand landmarks in real-time.

2️⃣ **Feature Extraction**
Coordinates are converted into relative positions (normalized).

3️⃣ **Model Prediction**

* Random Forest model predicts alphabet
* OR TensorFlow Lite model for optimized inference

4️⃣ **Output Display**
Predicted letter is shown on screen and can form words.

👉 This approach is commonly used in gesture recognition systems using landmark-based features ([GitHub][1])

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/Tathagata2022/Sign_Language_Detection_System.git
cd Sign_Language_Detection_System
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv asl_env
source asl_env/bin/activate        # Linux/Mac
asl_env\Scripts\activate           # Windows
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Step 1: Collect Dataset

```
python collect_data.py
```

* Press `a–z` → record gestures
* Space → pause
* `q` → exit

---

### 🔹 Step 2: Train Model

```
python train_model.py
```

* Trains Random Forest model
* Saves as `asl_model.pkl`

---

### 🔹 Step 3: Run Real-Time Detection

```
python main2.py
```

---

### 🔹 Step 4: Advanced Translator (Optional)

```
python translator.py
```

* Uses TensorFlow Lite for faster inference

---

## 📦 Requirements

All dependencies are listed in:

```
requirements.txt
```

---

## ⚠️ Important Notes

* Webcam is required 🎥
* Ensure model files exist:

  * `asl_model.pkl`
  * `keypoint_classifier.tflite` (for advanced mode)
* Good lighting improves accuracy
* Keep camera stable for better detection

---

## 📸 Future Improvements

* 🧠 Deep Learning (CNN/LSTM models)
* 📱 Mobile app deployment
* 🗣️ Speech output integration
* 🌍 Multi-language sign support
* ☁️ Cloud-based inference system

---

## 👨‍💻 Author

**Tathagata Mandal**

---

## 🌟 Project Highlights

✔️ Complete ML pipeline
✔️ Real-time inference system
✔️ Lightweight + efficient
✔️ Practical accessibility application

---

## 📜 License

This project is open-source and available for educational and research purposes.

[1]: https://github.com/MaitreeVaria/Indian-Sign-Language-Detection?utm_source=chatgpt.com "MaitreeVaria/Indian-Sign-Language-Detection: This project ..."

