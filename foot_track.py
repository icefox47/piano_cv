import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

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
        left_ankle_x = int(left_ankle.x * frame_width)
        left_ankle_y = int(left_ankle.y * frame_height)
        right_ankle_x = int(right_ankle.x * frame_width)
        right_ankle_y = int(right_ankle.y * frame_height)

        # Draw circles at the ankles for visualization
        cv2.circle(frame, (left_ankle_x, left_ankle_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (right_ankle_x, right_ankle_y), 5, (0, 255, 0), -1)

        # Show the coordinates on the frame
        cv2.putText(frame, f"Left Ankle: ({left_ankle_x}, {left_ankle_y})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Ankle: ({right_ankle_x}, {right_ankle_y})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with the detected pose and coordinates
    cv2.imshow('Foot Location Tracker', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

