import json
from Vehicle_Counting_in_Lanes import calculate_vehicle_density
import cv2

def save_density_to_json(frame):
    densities = calculate_vehicle_density(frame)

    # Write densities to JSON file
    with open('vehicle_densities.json', 'w') as f:
        json.dump(densities, f)

# Example usage
cap = cv2.VideoCapture('your_video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1920, 1080))

    # Save density data
    save_density_to_json(frame)

    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
