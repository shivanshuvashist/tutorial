import cv2
import numpy as np

# Open video file
cap = cv2.VideoCapture('/Users/shivanshuvashist/Downloads/RPReplay_Final1705842499.MP4')

# Define the lower and upper bounds of the cricket ball color (white)
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 50, 255])

# Create background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    
        # Apply thresholding to get binary image
    _, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)
    
    #fg_mask = bg_subtractor.apply(gray)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours
    for contour in contours:
        # Filter contours based on area or other criteria if needed
        area = cv2.contourArea(contour)
        if area > 2000:
            # Draw bounding box around detected object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('Tutorial', frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
