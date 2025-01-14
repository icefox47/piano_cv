import pygame
import time
import cv2
import mediapipe as mp

# Initialize pygame mixer
pygame.mixer.init()

# Define a function to play sound
def play_sound():
    # Load a sound (make sure to provide a valid path to a sound file on your system)
    sound = pygame.mixer.Sound('c4.mp3')  # Replace with the correct path to your sound file
    sound.play()

# Define the area to check (example values)
y_min = 200  # Minimum y value of the ankle area
y_max = 300  # Maximum y value of the ankle area

# List of captured ankle coordinates
ankle_data = [
    (245, 255), (245, 255), (245, 255), (246, 255), (246, 255), (246, 256),
    (246, 256), (246, 256), (246, 256), (246, 256), (246, 256), (246, 256),
    (246, 256), (246, 255), (246, 255), (246, 255), (246, 255), (246, 255),
    (246, 255), (246, 255), (246, 255), (246, 256), (246, 256), (246, 256),
    (246, 256), (247, 257), (249, 257), (250, 258), (250, 258), (250, 258),
    (243, 257), (237, 255), (237, 257), (248, 257), (263, 259), (270, 259),
    (272, 262), (273, 265), (270, 255), (270, 249), (271, 253), (271, 265),
    (273, 270), (274, 272), (273, 275), (272, 275), (272, 275), (273, 276),
    (274, 276), (273, 276), (273, 276), (273, 276), (273, 275), (273, 275),
    (273, 276), (273, 276), (273, 276), (273, 276), (274, 276), (274, 276),
    (274, 276), (272, 276), (260, 276), (253, 274), (255, 272), (270, 272),
    (274, 274), (278, 276), (278, 277), (277, 270), (277, 256), (276, 260),
    (276, 273), (276, 276), (276, 279), (276, 280), (276, 279), (276, 279),
    (276, 279), (276, 280), (276, 279)
]

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up webcam
cap = cv2.VideoCapture(2)

# Main loop to capture frames and process them
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the Pose model
    results = pose.process(rgb_frame)

    # Check if landmarks were detected
    if results.pose_landmarks:
        # Draw the pose landmarks on the frame (optional, for visualization)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the y-coordinates of the left and right ankles (Landmarks 29 and 30 for left and right ankles)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert normalized coordinates to pixel coordinates
        height, width, _ = frame.shape
        left_ankle_y = int(left_ankle.y * height)
        right_ankle_y = int(right_ankle.y * height)

        # Print ankle coordinates (optional)
        print(f"Left ankle y: {left_ankle_y}, Right ankle y: {right_ankle_y}")

        # Check if both ankles' y-values are within the defined area
        if y_min <= left_ankle_y <= y_max and y_min <= right_ankle_y <= y_max:
            # Check if the current ankle coordinates are in the previously captured data
            if (left_ankle_y, right_ankle_y) in ankle_data:
                print("Ankle coordinates match! Playing sound.")
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

