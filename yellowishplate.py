import cv2
import numpy as np

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

    # Define the lower and upper bounds for detecting the yellowish-green color in HSV space
    # Adjust these values to match the yellowish-green shade you're looking for
    lower_yellow_green = np.array([40, 60, 60])  # Lower bound of yellowish-green in HSV
    upper_yellow_green = np.array([80, 255, 255])  # Upper bound of yellowish-green in HSV

    # Create a mask for the yellowish-green regions in the frame
    mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

    # Find contours of the yellowish-green regions (which corresponds to the object)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box around the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the center of the bounding box
        object_center = (x + w // 2, y + h // 2)

        # Display the position of the object (center of the bounding box)
        cv2.putText(frame, f"Object Position: ({object_center[0]}, {object_center[1]})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the mask and the frame with object position
    cv2.imshow('Mask', mask)
    cv2.imshow('Object Position', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

