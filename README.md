# SignLanguageVowelAI
AI-Based ASL vowel recognition using hand landmarks

# Sign Language Vowel Recognition
This project detects ASL vowels (A, E, I, O, U) in real time using a webcam.

# How it works
- Uses MediaPipe to detect hand landmarks
- Applies simple rule-based logic
- Displays detected vowel on screen

# Technologies
- Python
- OpenCV
- MediaPipe

# Run
pip install opencv-python mediapipe
python predict_live.py

# Files
- `predict_live.py` – Real-time vowel recognition using MediaPipe (main file)
- `train_model.py` – CNN training script (future scope, not required to run live demo)

# Note
`train_model.py` is kept for future scope and is not required to run this project.

# Dataset (for Training)
The CNN model was trained using the "ASL Alphabet Dataset" from Kaggle.

Download link:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

⚠️ The dataset is NOT included in this repository due to size.
