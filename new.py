import pygame
import time
import cv2
import mediapipe as mp

# Initialize pygame mixer
pygame.mixer.init()

# Define a function to play sound
def play_sound():
    # Load a sound (make sure to provide a valid path to a sound file on your system)
    sound = pygame.mixer.Sound('c4.mp3')
    sound.play()

# Define the area to check (example values)
y_min = 200  # Minimum y value of the ankle area
y_max = 300  # Maximum y value of the ankle area

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV webcam capture
cap = cv2.VideoCapture(2)

# Main loop to capture frames and process them
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is not captured properly, break out of loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally (optional, for a mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB (as MediaPipe expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    # Check if landmarks were detected
    if results.pose_landmarks:
        # Draw the pose landmarks on the frame (optional, for visualization)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the y-coordinates of the left and right ankles (Landmarks 29 and 30 for left and right ankles)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]

        # Print ankle coordinates (optional)
        print(f"Left ankle y: {left_ankle}, Right ankle y: {right_ankle}")

        # Check if both ankles' y-values are within the defined area
        if y_min <= left_ankle <= y_max and y_min <= right_ankle <= y_max:
            play_sound()
            time.sleep(1)  # Wait 1 second before checking the next values

    # Display the captured frame in a window (optional)
    cv2.imshow("Webcam", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

