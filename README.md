# Emotion Mirror 🪞

A robust, real-time facial emotion recognition web application built with Python, Streamlit, OpenCV, and DeepFace.

## Features
- **Real-Time Analysis**: Captures webcam feed securely within the browser and analyzes expressions on the fly.
- **Edge-Case Handling**: 
  - Displays "No face detected" gently when no one is in frame.
  - Dynamically handles multiple faces, mapping bounding boxes and emotions to each individual separately.
- **Polished UI**: Styled with premium dark-mode aesthetics to look modern and professional.
- **Human-Like Error Catching**: Gracefully falls back to a "Scanning..." state if motion blur or awkward angles cause frame-level analysis failures.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system. 

### 2. Create Virtual Environment
It is strictly required to install dependencies inside a virtual environment to keep your system clean. Open your terminal in the project directory and run:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
*Note: The first time you run this, DeepFace will download the pre-trained weights for the emotion detection model. This only happens once and may take a few seconds depending on your internet connection.*

## What I Learned & Found Challenging
Building this application reinforced my understanding of bridging deep learning models with responsive user interfaces. The main challenge was ensuring the real-time processing loop remained performant. Running complex neural networks natively on every frame can be computationally heavy and cause lag. To solve this, I employed a hybrid approach: I used the lightweight OpenCV Haar Cascade classifier for high-speed face detection (bounding boxes), and then isolated those regions of interest (ROIs) to pass into DeepFace strictly for emotion classification. This ensured smooth video playback while retaining high-accuracy emotion prediction, directly addressing the requirement for a polished, functional experience rather than a slow prototype.

---
*Created for the Bipolar Factory assignment.*
