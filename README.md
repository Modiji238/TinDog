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
import mediapipe as mp
import numpy as np
import math

# Pose initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)

# Exponential Smoother
class EMAFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.last = None

    def smooth(self, value):
        if self.last is None:
            self.last = value
        else:
            self.last = self.alpha * value + (1 - self.alpha) * self.last
        return self.last

# Utility
def get_angle(a, b, c):
    """Returns the angle at point b given three 2D/3D points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Smoothers
flexion_smoother = EMAFilter(0.3)
tilt_smoother = EMAFilter(0.3)
rotation_smoother = EMAFilter(0.3)

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Key landmark points
        nose = [lm[mp_pose.PoseLandmark.NOSE].x * w, lm[mp_pose.PoseLandmark.NOSE].y * h]
        shoulder_l = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
        shoulder_r = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
        ear_l = [lm[mp_pose.PoseLandmark.LEFT_EAR].x * w, lm[mp_pose.PoseLandmark.LEFT_EAR].y * h]
        ear_r = [lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w, lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h]
        mid_shoulder = [(shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2]

        # Flexion/Extension (angle between torso vertical and nose vector)
        neck_angle = get_angle(shoulder_l, nose, shoulder_r)
        flexion = 180 - neck_angle  # Closer to 90° means full flexion
        flexion = flexion_smoother.smooth(flexion)

        # Lateral Tilt (difference in ears' Y)
        tilt_raw = ear_l[1] - ear_r[1]
        tilt_angle = np.clip(tilt_raw, -100, 100) / 2  # Scale to ~45 deg max
        tilt_angle = tilt_smoother.smooth(tilt_angle)

        # Rotation (difference in ears' X position)
        rot_raw = ear_l[0] - ear_r[0]
        rot_angle = np.clip(rot_raw, -200, 200) / 4  # scale
        rot_angle = rotation_smoother.smooth(rot_angle)

        # Display
        cv2.putText(frame, f"Flexion/Extension: {flexion:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Lateral Tilt: {tilt_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Rotation: {rot_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("BlazePose - Cervical ROM", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
