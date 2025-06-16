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




import cv2
import mediapipe as mp
import numpy as np

# Setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(model_complexity=1)
face = mp_face.FaceMesh(refine_landmarks=True)

# EMA Filter
class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last = None
    def smooth(self, value):
        if self.last is None:
            self.last = value
        else:
            self.last = self.alpha * value + (1 - self.alpha) * self.last
        return self.last

pitch_filter = EMAFilter(0.3)
yaw_filter = EMAFilter(0.3)
roll_filter = EMAFilter(0.3)

# 3D model points (generic reference)
model_points = np.array([
    (0.0, 0.0, 0.0),              # Nose tip
    (0.0, -330.0, -65.0),         # Chin
    (-225.0, 170.0, -135.0),      # Left eye corner
    (225.0, 170.0, -135.0),       # Right eye corner
    (-150.0, -150.0, -125.0),     # Left mouth corner
    (150.0, -150.0, -125.0)       # Right mouth corner
], dtype=np.float64)

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose and face mesh
    pose_result = pose.process(rgb)
    face_result = face.process(rgb)

    if pose_result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if face_result.multi_face_landmarks:
        lm = face_result.multi_face_landmarks[0].landmark

        image_points = np.array([
            (lm[1].x * w, lm[1].y * h),     # Nose tip
            (lm[152].x * w, lm[152].y * h), # Chin
            (lm[263].x * w, lm[263].y * h), # Right eye
            (lm[33].x * w, lm[33].y * h),   # Left eye
            (lm[287].x * w, lm[287].y * h), # Right mouth
            (lm[57].x * w, lm[57].y * h)    # Left mouth
        ], dtype=np.float64)

        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP for head pose
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        rot_mat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rot_mat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [float(a) for a in euler_angles]

        # Smooth
        pitch = pitch_filter.smooth(pitch)
        yaw = yaw_filter.smooth(yaw)
        roll = roll_filter.smooth(roll)

        # Display
        cv2.putText(frame, f"Flexion/Extension (Pitch): {pitch:.1f}°", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Rotation (Yaw): {yaw:.1f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"Lateral Tilt (Roll): {roll:.1f}°", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Hybrid Accurate cROM Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

