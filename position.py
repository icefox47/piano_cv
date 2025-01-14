import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

# Define the fixed position where the object is located (in pixels)
# You can set this to a specific y-coordinate in the frame.
object_y_position = 300  # For example, 300 pixels from the top

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

        # Check if both ankles are above the object
        if left_ankle_y < object_y_position and right_ankle_y < object_y_position:
            cv2.putText(frame, "Legs on object!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Save the frame when condition is met
            cv2.imwrite('legs_on_object.jpg', frame)
            print("Captured image: Legs on object")

    # Display the resulting frame
    cv2.imshow('Body Movement Detection with Fixed Object Position', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

