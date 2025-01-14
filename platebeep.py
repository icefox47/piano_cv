import cv2
import numpy as np
import mediapipe as mp
import pygame  # For playing sound

# Initialize pygame mixer for sound
pygame.mixer.init()

# Function to play custom beep sound
def beep():
    """Function to play the custom beep sound."""
    pygame.mixer.Sound('c4.mp3').play()  # Ensure you have the 'beep_sound.wav' file in the directory

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

# Define the color range for yellowish-green in HSV
lower_yellowish_green = np.array([40, 100, 100])  # HSV lower bound
upper_yellowish_green = np.array([80, 255, 255])  # HSV upper bound

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for yellowish-green color
    mask = cv2.inRange(hsv_frame, lower_yellowish_green, upper_yellowish_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour is found (yellowish-green object detected)
    object_detected = False
    object_x, object_y, object_w, object_h = 0, 0, 0, 0

    if contours:
        object_detected = True

        # Draw bounding box around the detected yellowish-green object
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Only consider large enough contours (adjust area threshold)
                x, y, w, h = cv2.boundingRect(contour)
                object_x, object_y, object_w, object_h = x, y, w, h
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Yellowish Green Object Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        left_ankle_x = int(left_ankle.x * frame_width)
        left_ankle_y = int(left_ankle.y * frame_height)
        right_ankle_x = int(right_ankle.x * frame_width)
        right_ankle_y = int(right_ankle.y * frame_height)

        # Draw circles at the ankles for visualization
        cv2.circle(frame, (left_ankle_x, left_ankle_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (right_ankle_x, right_ankle_y), 5, (0, 255, 0), -1)

        # Check if any foot is on top of the yellowish-green object
        if object_detected:
            if (object_x <= left_ankle_x <= object_x + object_w and 
                object_y <= left_ankle_y <= object_y + object_h) or \
               (object_x <= right_ankle_x <= object_x + object_w and 
                object_y <= right_ankle_y <= object_y + object_h):
                
                # If feet are detected on top of the object, play the sound
                beep()
                cv2.putText(frame, "Foot on Object - Beep!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with bounding boxes and detection message
    cv2.imshow('Object and Foot Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

