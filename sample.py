import cv2
import numpy as np

# Function to detect traffic signal light color
def detect_traffic_signal_color(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color ranges for traffic signal lights
    # You may need to adjust these ranges based on the specific conditions and lighting of your image
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])

    # Create a mask for red and green colors
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Apply some morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the masks
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check which color has more significant contours
    if len(red_contours) > len(green_contours):
        return "Red"
    elif len(green_contours) > len(red_contours):
        return "Green"
    else:
        return "No Signal"

# Load the image
image = cv2.imread('download.png')

# Detect the traffic signal color
color = detect_traffic_signal_color(image)

# Display the color
print(f"Traffic Signal Color: {color}")

# Optionally, you can display the image with the color text overlay
cv2.putText(image, f"Traffic Signal: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Traffic Signal Color Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
