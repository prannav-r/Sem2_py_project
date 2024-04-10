import cv2
import numpy as np

# Function to detect jump using background subtraction
def detect_jump(frame, background):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction
    diff = cv2.absdiff(background, gray)
    
    # Apply threshold to highlight differences
    _, threshold = cv2.threshold(diff, 3, 1440, cv2.THRESH_BINARY)
    
    # Perform morphological operations to remove noise
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    threshold = cv2.dilate(threshold, None, iterations=2)
    
    # Find contours of the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables for tracking the lowest contour and its height
    lowest_contour = None
    lowest_height = float('inf')
    
    # Find the lowest contour representing the person's body
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust the threshold as needed
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Update lowest contour if the current contour is lower than the previous lowest contour
            if y + h < lowest_height:
                lowest_contour = contour
                lowest_height = y + h
    
    # Check if the lowest contour is above a certain threshold from the bottom of the frame
    if lowest_contour is not None and lowest_height < frame.shape[0] - 200:  # Adjust the threshold as needed
        return True
    
    return False

# Open video capture
cap = cv2.VideoCapture("D:/Github/Sem2_py_project/4.mp4") # Replace 'your_video.mp4' with the path to your video file

# Read the first frame to initialize background
_, background = cap.read()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Define kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect jumps using background subtraction
    jump = detect_jump(frame, background)
    
    # Display result on the frame
    if jump:
        cv2.putText(frame, 'Jump not Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame,"Jump Detected",(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)        
    
    # Display frame
    cv2.imshow('Frame', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()