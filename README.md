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
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from PIL import Image
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== Load FSA-Net Keras model ==========
model = load_model("fsanet_capsule.h5", compile=False)

# ========== Constants ==========
idx_tensor = np.arange(66, dtype=np.float32)
plot_len = 100  # length of rolling plot
frame_size = 64  # expected input size for model

# ========== MediaPipe face detection ==========
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# ========== Plot setup ==========
plt.style.use("ggplot")
fig, ax = plt.subplots()
x_vals = list(range(plot_len))
yaw_line, = ax.plot(x_vals, [0]*plot_len, label="Yaw")
pitch_line, = ax.plot(x_vals, [0]*plot_len, label="Pitch")
roll_line, = ax.plot(x_vals, [0]*plot_len, label="Roll")
ax.set_ylim(-60, 60)
ax.legend(loc="upper right")

yaw_data, pitch_data, roll_data = [], [], []
yaw_smooth, pitch_smooth, roll_smooth = deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)

# ========== Baseline for neutral ==========
baseline_yaw = baseline_pitch = baseline_roll = None

# ========== Start webcam ==========
cap = cv2.VideoCapture(0)

def update(frame_id):
    global baseline_yaw, baseline_pitch, baseline_roll

    ret, frame = cap.read()
    if not ret:
        return

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)

    if results.detections:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Crop and preprocess face
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + bw, w), min(y + bh, h)
        face_img = frame_rgb[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (frame_size, frame_size))
        face_img = face_img.astype(np.float32) / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        preds = model.predict(face_img)[0]
        yaw = np.sum(tf.nn.softmax(preds[0:66]) * idx_tensor) * 3 - 99
        pitch = np.sum(tf.nn.softmax(preds[66:132]) * idx_tensor) * 3 - 99
        roll = np.sum(tf.nn.softmax(preds[132:198]) * idx_tensor) * 3 - 99

        # Smooth values
        yaw_smooth.append(yaw)
        pitch_smooth.append(pitch)
        roll_smooth.append(roll)
        yaw_avg = np.mean(yaw_smooth)
        pitch_avg = np.mean(pitch_smooth)
        roll_avg = np.mean(roll_smooth)

        # Calibrate neutral
        if baseline_yaw is None:
            baseline_yaw = yaw_avg
            baseline_pitch = pitch_avg
            baseline_roll = roll_avg

        # Compute deviation
        dyaw = yaw_avg - baseline_yaw
        dpitch = pitch_avg - baseline_pitch
        droll = roll_avg - baseline_roll

        # Update data for plot
        yaw_data.append(dyaw)
        pitch_data.append(dpitch)
        roll_data.append(droll)

        if len(yaw_data) > plot_len:
            yaw_data.pop(0)
            pitch_data.pop(0)
            roll_data.pop(0)

        # Update plot
        yaw_line.set_ydata(yaw_data + [0]*(plot_len - len(yaw_data)))
        pitch_line.set_ydata(pitch_data + [0]*(plot_len - len(pitch_data)))
        roll_line.set_ydata(roll_data + [0]*(plot_len - len(roll_data)))

        # Display text
        text = f"Yaw: {dyaw:+.1f}° | Pitch: {dpitch:+.1f}° | Roll: {droll:+.1f}°"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2)

    cv2.imshow("FSA-Net Keras Live Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

    return yaw_line, pitch_line, roll_line

ani = FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()
