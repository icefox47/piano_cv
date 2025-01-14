import cv2
import mediapipe as mp

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set up webcam
cap = cv2.VideoCapture(2)

def draw_pose_landmarks(frame, results):
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Access specific landmarks (e.g., left ankle, right ankle)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Convert normalized coordinates to pixel coordinates
        height, width, _ = frame.shape
        left_ankle_y = int(left_ankle.y * height)
        right_ankle_y = int(right_ankle.y * height)
        
        # Print the y-coordinates of the ankles (for debugging)
        print(f"Left Ankle y: {left_ankle_y}, Right Ankle y: {right_ankle_y}")
        
        # Draw circles on the ankles
        cv2.circle(frame, (int(left_ankle.x * width), left_ankle_y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_ankle.x * width), right_ankle_y), 10, (255, 0, 0), -1)
        
    return frame

def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with the Pose model
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks and print ankle coordinates
        frame = draw_pose_landmarks(frame, results)
        
        # Display the video feed with landmarks
        cv2.imshow("Pose Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

