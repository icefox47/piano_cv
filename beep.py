import cv2
import numpy as np
import mediapipe as mp
import pygame  # For cross-platform sound

# Initialize pygame mixer for sound
pygame.mixer.init()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

# Define the coordinate range for the foot position
x_range = (413, 414)  # x-coordinate should be between 413 and 414
y_range = (383, 384)  # y-coordinate should be between 383 and 384

# Create a directory to store captured images (if not exists)
import os
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Initialize counter for image capture
capture_count = 0

def beep():
    """Function to play a beep sound."""
    pygame.mixer.Sound(pygame.mixer.Sound('beep_sound.wav')).play()  # Ensure you have a beep sound file

def capture_image(frame, count):
    """Function to capture and save an image."""
    filename = f"captured_images/foot_on_position_{count}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Captured image: {filename}")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the positions of the left and right ankles
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert the normalized coordinates to pixel values
        frame_height, frame_width, _ = frame.shape
        left_ankle_y = int(left_ankle.y * frame_height)
        right_ankle_y = int(right_ankle.y * frame_height)
        left_ankle_x = int(left_ankle.x * frame_width)
        right_ankle_x = int(right_ankle.x * frame_width)

        # Draw circles at the ankles for visualization
        cv2.circle(frame, (left_ankle_x, left_ankle_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (right_ankle_x, right_ankle_y), 5, (0, 255, 0), -1)

        # Check if either ankle is within the defined coordinate range
        if (x_range[0] <= left_ankle_x <= x_range[1] and y_range[0] <= left_ankle_y <= y_range[1]) or \
           (x_range[0] <= right_ankle_x <= x_range[1] and y_range[0] <= right_ankle_y <= y_range[1]):
            # If either ankle is in the specified range, play beep sound and capture the image
            beep()
            capture_count += 1
            capture_image(frame, capture_count)

    # Display the frame with the detected pose
    cv2.imshow('Body Movement Detection - Beep and Image Capture', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

