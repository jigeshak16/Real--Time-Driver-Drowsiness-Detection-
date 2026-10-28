# 🚗 Real Time – Driver Drowsiness Detection System

It is an AI-powered driver safety project designed to detect signs of driver drowsiness and reduced vigilance using video analysis and real-time webcam monitoring.

## ✨ Features

- **Live Monitoring:** Monitors eye closure through webcam input.
- **CNN-LSTM Model:** Analyses video sequences to classify vigilance states.
- **Temporal Attention:** Helps the model focus on important frames across a video sequence.
- **Multi-Agent System:** Monitor, Decision, and Alert Agents work together to detect prolonged eye closure and trigger warnings.
- **Streamlit Dashboard:** Provides live monitoring, dataset testing, model evaluation, and incident history.

## 🧠 Vigilance Classes

The video classification model predicts three classes:

- Alert
- Low Vigilant
- Drowsy

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Streamlit

## 🏗️ Model Architecture

**Video Frames → MobileNetV2 CNN → LSTM → Temporal Attention → Softmax Classification**

The CNN extracts visual features from video frames, the LSTM learns temporal patterns, and the attention layer helps the model focus on important frames before classification.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jigeshak16/Real--Time-Driver-Drowsiness-Detection-.git
cd Real--Time-Driver-Drowsiness-Detection-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## 📊 Model Evaluation

The initial CNN-LSTM model achieved **40% accuracy** on a held-out test set of 15 videos. Evaluation showed that the model struggled particularly with the Low Vigilant class.

These results are preliminary. Further training, data preprocessing, hyperparameter tuning, and testing on additional unseen videos are needed to improve performance.

## 🔮 Future Improvements

- Optimise processing of the large UTA-RLDD dataset.
- Improve classification of Low Vigilant and Drowsy states.
- Experiment with different learning rates, epochs, and sequence lengths.
- Evaluate performance on more unseen subjects and videos.
- Further integrate video classification with live monitoring.

## ⚠️ Disclaimer

AEGIS is an experimental academic project. Its predictions have not been sufficiently validated for real-world driving safety and should not replace established driver-safety systems.

## 👩‍💻 Authors

**Jigesha Kapoor**  
**Ruchi Khairnar**

Artificial Intelligence and Machine Learning Students

---

*Developed as an academic project on AI-powered driver safety and drowsiness detection.*
