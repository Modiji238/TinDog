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
import pandas as pd

# Initialize MediaPipe BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils

# List of landmark indices you want to save (change as needed)
# Example: Nose = 0, Left shoulder = 11, Right shoulder = 12
selected_landmarks = [0, 11, 12]

# Initialize CSV storage
all_data = []

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' to use a video file

frame_num = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image and get pose landmarks
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        frame_data = {"frame": frame_num}
        for idx in selected_landmarks:
            landmark = results.pose_landmarks.landmark[idx]
            frame_data[f"{mp_pose.PoseLandmark(idx).name}_x"] = landmark.x
            frame_data[f"{mp_pose.PoseLandmark(idx).name}_y"] = landmark.y
            frame_data[f"{mp_pose.PoseLandmark(idx).name}_z"] = landmark.z
            frame_data[f"{mp_pose.PoseLandmark(idx).name}_vis"] = landmark.visibility
        all_data.append(frame_data)

        # Optionally draw the pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show video stream
    cv2.imshow('MediaPipe BlazePose', frame)
    frame_num += 1

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save data to CSV
df = pd.DataFrame(all_data)
df.to_csv("selected_landmarks.csv", index=False)

print("Saved landmark coordinates to selected_landmarks.csv")
