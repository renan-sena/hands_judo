import cv2
import mediapipe as mp
import time
import numpy as np
from colorama import Fore, Back, Style

# MediaPipe hand detection configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Control variables
blue_grip = False
white_grip = False
blue_start_time = 0
white_start_time = 0
blue_disappear_time = 0
white_disappear_time = 0
disappearance_tolerance = 2 
winner = None 

# Kalman Filter variables for smoothing and predicting movements
kalman_filter_blue = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements (x, y)
kalman_filter_white = cv2.KalmanFilter(4, 2)

# Initialize Kalman Filter parameters
def initialize_kalman(kf):
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

initialize_kalman(kalman_filter_blue)
initialize_kalman(kalman_filter_white)

# Function to update the Kalman Filter
def update_kalman(kf, x, y):
    measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kf.correct(measurement)
    prediction = kf.predict()
    return int(prediction[0]), int(prediction[1])

# Function to check if the grip is active for more than 7 seconds
def check_grip(start_time, current_time):
    return (current_time - start_time) >= 7

# Function to draw grip status on the screen
def draw_status(frame, blue_grip, white_grip):
    blue_text = f'Blue Grip: {"Active" if blue_grip else "Inactive"}'
    white_text = f'White Grip: {"Active" if white_grip else "Inactive"}'
    cv2.putText(frame, blue_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, white_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Function to detect the colors of the kimonos (blue and white)
def detect_kimono_color(frame_hsv, frame):
    # Defining color ranges in HSV for blue and white
    blue_low = np.array([100, 150, 50])
    blue_high = np.array([140, 255, 255])

    white_low = np.array([0, 0, 200])
    white_high = np.array([180, 30, 255])

    # Masks for blue and white colors
    blue_mask = cv2.inRange(frame_hsv, blue_low, blue_high)
    white_mask = cv2.inRange(frame_hsv, white_low, white_high)

    # Check if large areas of blue or white are present in the frame
    blue_athlete = cv2.countNonZero(blue_mask) > 1000
    white_athlete = cv2.countNonZero(white_mask) > 1000

    return blue_athlete, white_athlete

# Open video
video_path = 'video_path.mp4' 
cap = cv2.VideoCapture(video_path)

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error opening video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    
    # Convert to RGB and HSV
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect kimono color
    blue_athlete, white_athlete = detect_kimono_color(frame_hsv, frame)
    
    # Process the frame for hand detection
    results = hands.process(frame_rgb)

    # Get the current time
    current_time = time.time()

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Getting the wrist point (landmark 9) to determine position
            wrist_x = int(hand_landmarks.landmark[9].x * frame.shape[1])
            wrist_y = int(hand_landmarks.landmark[9].y * frame.shape[0])

            # Draw hand landmarks
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the hand belongs to the blue or white athlete
            if blue_athlete:
                blue_grip = True
                blue_disappear_time = 0  # Reset disappearance counter
                blue_start_time = blue_start_time or current_time  # Mark the start time of the grip

                # Update Kalman Filter with the current position
                pred_blue_x, pred_blue_y = update_kalman(kalman_filter_blue, wrist_x, wrist_y)

            elif white_athlete:
                white_grip = True
                white_disappear_time = 0  # Reset disappearance counter
                white_start_time = white_start_time or current_time  # Mark the start time of the grip

                # Update Kalman Filter with the current position
                pred_white_x, pred_white_y = update_kalman(kalman_filter_white, wrist_x, wrist_y)

    else:
        # No hands detected: start disappearance counter
        if blue_grip:
            blue_disappear_time += 1/30  # Assuming video has 30fps
            if blue_disappear_time > disappearance_tolerance:
                blue_grip = False
                blue_start_time = 0

        if white_grip:
            white_disappear_time += 1/30
            if white_disappear_time > disappearance_tolerance:
                white_grip = False
                white_start_time = 0

    # Check if the grip has been maintained for 7 seconds
    if check_grip(blue_start_time, current_time) and winner is None:
        print(Fore.BLUE + "Blue athlete won the grip!")
        winner = "blue"  # Mark that the blue athlete won

    if check_grip(white_start_time, current_time) and winner is None:
        print(Fore.RED + "White athlete won the grip!")
        winner = "white"  # Mark that the white athlete won

    # Draw grip status on the frame
    draw_status(frame, blue_grip, white_grip)

    # Show the video
    cv2.imshow('Judo Grip Tracking', frame)

    # Add delay between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    