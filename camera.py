import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import datetime
from collections import deque
import numpy as np
import time

keyboard = Controller()
cap = cv2.VideoCapture(2)  # camera index

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

steering = "neutral"
kick_ready = True

# Sensitivity thresholds
KNEE_DIFF_THRESHOLD = 0.012
ARM_EXTENSION_THRESHOLD = 0.05
KICK_FORWARD_THRESHOLD = 0.01

# Smoothing history
knee_history = deque(maxlen=5)
left_wrist_history = deque(maxlen=5)
right_wrist_history = deque(maxlen=5)

# For continuous accelerate
ACCEL_RELEASE_FRAMES = 30
stop_counter = 0

def log_action(action):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {action}")

accelerating = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Accelerate ---
        left_knee_y = lm[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE].y
        knee_diff = abs(left_knee_y - right_knee_y)
        knee_history.append(knee_diff)
        knee_diff_avg = np.mean(knee_history)

        if knee_diff_avg > KNEE_DIFF_THRESHOLD:
            if not accelerating:
                keyboard.press(Key.space)
                accelerating = True
                log_action("Start Accelerating")
            stop_counter = 0
        else:
            stop_counter += 1
            if accelerating and stop_counter >= ACCEL_RELEASE_FRAMES:
                keyboard.release(Key.space)
                accelerating = False
                log_action("Stop Accelerating")

        # --- Steering (gradual) ---
        left_wrist_x = lm[mp_pose.PoseLandmark.LEFT_WRIST].x
        right_wrist_x = lm[mp_pose.PoseLandmark.RIGHT_WRIST].x
        left_shoulder_x = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        right_shoulder_x = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

        left_wrist_history.append(left_wrist_x)
        right_wrist_history.append(right_wrist_x)
        left_avg = np.mean(left_wrist_history)
        right_avg = np.mean(right_wrist_history)

        # Determine analog steering value (-1 = full left, 1 = full right)
        left_offset = left_avg - left_shoulder_x
        right_offset = right_avg - right_shoulder_x

        steering_value = 0
        if left_offset < -ARM_EXTENSION_THRESHOLD:
            steering_value = min(1.0, -left_offset * 5)  # scale factor for sensitivity
        elif right_offset > ARM_EXTENSION_THRESHOLD:
            steering_value = max(-1.0, -right_offset * 5)

        # Map analog value to discrete keys
        if steering_value > 0.1:  # turn right
            if steering != "right":
                if steering == "left":
                    keyboard.release("a")
                keyboard.press("d")
                steering = "right"
                log_action("Steering RIGHT")
        elif steering_value < -0.1:  # turn left
            if steering != "left":
                if steering == "right":
                    keyboard.release("d")
                keyboard.press("a")
                steering = "left"
                log_action("Steering LEFT")
        else:  # neutral
            if steering != "neutral":
                if steering == "right":
                    keyboard.release("d")
                elif steering == "left":
                    keyboard.release("a")
                steering = "neutral"
                log_action("Steering NEUTRAL")

        # --- Kick detection ---
        left_ankle_z = lm[mp_pose.PoseLandmark.LEFT_ANKLE].z
        right_ankle_z = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].z

        if kick_ready:
            if left_ankle_z < -KICK_FORWARD_THRESHOLD or right_ankle_z < -KICK_FORWARD_THRESHOLD:
                keyboard.press("q")
                time.sleep(0.05)  # hold key for 50ms so game registers
                keyboard.release("q")
                log_action("Kick! Ability Fired")
                kick_ready = False
        else:
            # Reset kick if either foot returns near neutral
            if abs(left_ankle_z) < 0.05 or abs(right_ankle_z) < 0.05:
                kick_ready = True

    # --- Visualization overlays ---
    status_text = f"Steering: {steering.upper()} | Accelerating: {'Yes' if accelerating else 'No'} | Kick Ready: {'Yes' if kick_ready else 'No'}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Mario Kart Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()

