import cv2
import mediapipe as mp
import numpy as np

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

    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for detecting the red bucket (HSV color space)
    # Adjust these values to match the shade of red of the bucket
    lower_red = np.array([0, 120, 70])  # Lower bound of red in HSV
    upper_red = np.array([10, 255, 255])  # Upper bound of red in HSV

    # Mask the frame to get only the red parts (bucket)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours of the red object (bucket)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_y_position = None  # Will be determined later based on the bucket's position

    if contours:
        # Find the largest contour which corresponds to the bucket
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center of the bounding box (for simplicity)
        object_y_position = y + h // 2  # Use the vertical center of the bucket as the object position

        # Optionally, display the bounding box around the bucket
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Bucket Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Now, perform pose detection to track body movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the positions of the left and right ankles
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert the normalized coordinates to pixel values
        frame_height, frame_width, _ = frame.shape
        left_ankle_y = int(left_ankle.y * frame_height)
        right_ankle_y = int(right_ankle.y * frame_height)

        # If the object is detected (bucket) and the legs are above the object, capture the image
        if object_y_position and left_ankle_y < object_y_position and right_ankle_y < object_y_position:
            cv2.putText(frame, "Legs on bucket!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Save the frame when condition is met
            cv2.imwrite('legs_on_bucket.jpg', frame)
            print("Captured image: Legs on bucket")

    # Display the resulting frame
    cv2.imshow('Body Movement Detection with Bucket Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

