                                        Jump Detection using python

Group Members:

.Tharun TV - 23011102109
.Prannav R - 23011102065
.Priyajit Biswal - 23011102068
.Rishab Rajeev - 23011102072

Jump Detection in Videos:

This Python script detects jumps in a video using background subtraction.

Requirements:

OpenCV library (https://opencv.org/)
NumPy library (https://numpy.org/)

Working:

The script performs the following steps:
1.	Imports libraries: Imports OpenCV (cv2) and NumPy (np) libraries.
2.	Defines detect_jump function: This function takes a frame and background image as input and performs the following:
    o	Converts the frame to grayscale.
    o	Applies background subtraction to highlight differences between the frame and background.
    o	Applies thresholding to create a binary image.
    o	Performs morphological operations (opening and dilation) to remove noise.
    o	Finds contours in the thresholded image.
    o	Identifies the lowest contour (likely representing the person's body) and its height.
    o	Checks if the lowest contour is above a certain threshold from the bottom of the frame (indicating a jump).
    o	Returns True if a jump is detected, False otherwise.
3.	Opens video capture: Opens the video using OpenCV's VideoCapture.
4.	Reads first frame for background: Reads the first frame and converts it to grayscale for background initialization.
5.	Defines kernel for morphological operations: Creates a kernel (used for noise removal) using NumPy.
6.	Loops through frames: Iterates through each frame in the video.
    o	Detects jump using the detect_jump function.
    o	Displays "Jump Detected" or "Jump Not Detected" on the frame based on the result.
7.	Shows video with results: Displays the video with jump detection labels on each frame.
8.	Breaks loop on 'q' press: Exits the loop when the user presses the 'q' key.
9.	Releases resources: Releases the video capture and closes all windows.

Graphs:

Accuracy Graph (pre-trained model)-

![alt text](8Ln9V.png)