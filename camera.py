import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import datetime
from collections import deque
import numpy as np

keyboard = Controller()
cap = cv2.VideoCapture(2)  # camera index

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

accelerating = False
steering = "neutral"
kick_ready = True

# Sensitivity thresholds (lower = more sensitive)
KNEE_DIFF_THRESHOLD = 0.05       # was 0.08
ARM_EXTENSION_THRESHOLD = 0.12   # was 0.20
KICK_FORWARD_THRESHOLD = 0.10    # was 0.15

# Smoothing history
knee_history = deque(maxlen=5)
left_wrist_history = deque(maxlen=5)
right_wrist_history = deque(maxlen=5)

def log_action(action):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {action}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # --- Accelerate ---
        left_knee_y = lm[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE].y
        knee_diff = abs(left_knee_y - right_knee_y)
        knee_history.append(knee_diff)
        knee_diff_avg = np.mean(knee_history)  # smooth over 5 frames

        if knee_diff_avg > KNEE_DIFF_THRESHOLD and not accelerating:
            keyboard.press(Key.space)
            accelerating = True
            log_action("Start Accelerating")
        elif knee_diff_avg <= KNEE_DIFF_THRESHOLD and accelerating:
            keyboard.release(Key.space)
            accelerating = False
            log_action("Stop Accelerating")

        # --- Steering ---
        left_wrist_x = lm[mp_pose.PoseLandmark.LEFT_WRIST].x
        right_wrist_x = lm[mp_pose.PoseLandmark.RIGHT_WRIST].x
        left_shoulder_x = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        right_shoulder_x = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

        left_wrist_history.append(left_wrist_x)
        right_wrist_history.append(right_wrist_x)
        left_avg = np.mean(left_wrist_history)
        right_avg = np.mean(right_wrist_history)

        new_steering = "neutral"
        if left_avg < left_shoulder_x - ARM_EXTENSION_THRESHOLD:
            new_steering = "left"
        elif right_avg > right_shoulder_x + ARM_EXTENSION_THRESHOLD:
            new_steering = "right"

        if new_steering != steering:
            if steering == "left":
                keyboard.release("a")
            elif steering == "right":
                keyboard.release("d")

            if new_steering == "left":
                keyboard.press("a")
            elif new_steering == "right":
                keyboard.press("d")

            steering = new_steering
            log_action(f"Steering {steering.upper()}")

        # --- Kick detection ---
        left_ankle_z = lm[mp_pose.PoseLandmark.LEFT_ANKLE].z
        right_ankle_z = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].z

        if kick_ready:
            if left_ankle_z < -KICK_FORWARD_THRESHOLD or right_ankle_z < -KICK_FORWARD_THRESHOLD:
                keyboard.press("q")
                keyboard.release("q")
                log_action("Kick! Ability Fired")
                kick_ready = False
        else:
            if abs(left_ankle_z) < 0.05 and abs(right_ankle_z) < 0.05:
                kick_ready = True

    cv2.imshow("Mario Kart Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()

