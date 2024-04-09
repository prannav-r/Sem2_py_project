import cv2
import numpy as np

# Function to detect jump using background subtraction
def detect_jump(frame, background):
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
cap = cv2.VideoCapture("D:/Github/py_project/3.mp4")  # Replace with the path to your video file

# Read the first frame to initialize background
_, background = cap.read()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Define kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Ground truth data (dummy example)
ground_truth = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]  # Dummy ground truth data (0: no jump, 1: jump)

# Initialize variables for accuracy calculation
total_frames = len(ground_truth)
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Loop through frames
for frame_number in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect jumps using background subtraction
    jump_detected = detect_jump(frame, background)
    
    # Compare with ground truth
    if jump_detected == 1 and ground_truth[frame_number] == 1:
        true_positives += 1
    elif jump_detected == 1 and ground_truth[frame_number] == 0:
        false_positives += 1
    elif jump_detected == 0 and ground_truth[frame_number] == 0:
        true_negatives += 1
    elif jump_detected == 0 and ground_truth[frame_number] == 1:
        false_negatives += 1

# Calculate accuracy metrics
accuracy = (true_positives + true_negatives) / total_frames
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print accuracy metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

# Release video capture
cap.release()