# TinDog
Tinder For Dogs-

1)Responsive Design:
  -Ensure the site is fully responsive and works well on different devices (mobiles, tablets, desktops).
  -Use Bootstrap's grid system and responsive utilities.
  
2)Consistent and Modern UI:
  -Use modern and clean design principles.
  -Ensure consistency in font sizes, colors, and spacing.
  -Use a cohesive color scheme that is appealing and appropriate for the theme.

Includes features such as hero sections,attractive and responsive buttons,a pricing table and a Carousel for brandings


import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import joblib

# === Load Trained Model and Label Encoder ===
model = load_model('lstm_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

SEQUENCE_LENGTH = 50
buffer = deque(maxlen=SEQUENCE_LENGTH)

# === Initialize MediaPipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# === Start Webcam ===
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
predicted_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get pose landmarks
    result = pose.process(rgb)

    if result.pose_landmarks:
        # Get nose landmark
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_coord = [nose.x, nose.y, nose.z]
        buffer.append(nose_coord)

        # Draw landmark
        h, w = frame.shape[:2]
        cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (0, 255, 0), -1)

        # Predict when buffer is full
        if len(buffer) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(buffer, axis=0)  # shape: (1, 50, 3)
            probs = model.predict(input_seq, verbose=0)[0]
            pred_index = np.argmax(probs)
            confidence = probs[pred_index]
            predicted_label = label_encoder.inverse_transform([pred_index])[0] if confidence > 0.8 else "Uncertain"

    # === Display prediction ===
    cv2.putText(frame, f'Prediction: {predicted_label}', (20, 40), font, 0.8, (0, 0, 255), 2)

    cv2.imshow("Exercise Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
