import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *
import time

video_path = 'vid_one.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

road_zoneA = np.array([[308, 789], [711, 807], [694, 492], [415, 492], [309, 790]], np.int32)
road_zoneB = np.array([[727, 797], [1123, 812], [1001, 516], [741, 525], [730, 795]], np.int32)
road_zoneC = np.array([[1116, 701], [1533, 581], [1236, 367], [1009, 442], [1122, 698]], np.int32)

zoneA_Line = np.array([road_zoneA[0], road_zoneA[1]]).reshape(-1)
zoneB_Line = np.array([road_zoneB[0], road_zoneB[1]]).reshape(-1)
zoneC_Line = np.array([road_zoneC[0], road_zoneC[1]]).reshape(-1)

tracker = Sort()
zoneAcounter = []
zoneBcounter = []
zoneCcounter = []

# Define max green light time
max_green_time = 120  # in seconds
update_interval = 10  # Interval to update signal timing in seconds
last_update_time = time.time()

def update_signal_timing(lane, signal_time):
    print(f"Updated signal timing for {lane}: {signal_time:.2f} seconds")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1920, 1080))
    results = model(frame)
    current_detections = np.empty([0, 5])

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_detect in ['car', 'truck', 'bus'] and conf > 60:
                detections = np.array([x1, y1, x2, y2, conf])
                current_detections = np.vstack([current_detections, detections])

    cv2.polylines(frame, [road_zoneA], isClosed=False, color=(0, 0, 255), thickness=8)
    cv2.polylines(frame, [road_zoneB], isClosed=False, color=(0, 255, 255), thickness=8)
    cv2.polylines(frame, [road_zoneC], isClosed=False, color=(255, 0, 0), thickness=8)

    track_results = tracker.update(current_detections)
    for result in track_results:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 - 40

        if zoneA_Line[0] < cx < zoneA_Line[2] and zoneA_Line[1] - 20 < cy < zoneA_Line[1] + 20:
            if id not in zoneAcounter:
                zoneAcounter.append(id)

        if zoneB_Line[0] < cx < zoneB_Line[2] and zoneB_Line[1] - 20 < cy < zoneB_Line[1] + 20:
            if id not in zoneBcounter:
                zoneBcounter.append(id)

        if zoneC_Line[0] < cx < zoneC_Line[2] and zoneC_Line[1] - 20 < cy < zoneC_Line[1] + 20:
            if id not in zoneCcounter:
                zoneCcounter.append(id)

    # Display vehicle counts
    cvzone.putTextRect(frame, f'LANE A Vehicles = {len(zoneAcounter)}', [1000, 99], thickness=4, scale=2.3, border=2)
    cvzone.putTextRect(frame, f'LANE B Vehicles = {len(zoneBcounter)}', [1000, 140], thickness=4, scale=2.3, border=2)
    cvzone.putTextRect(frame, f'LANE C Vehicles = {len(zoneCcounter)}', [1000, 180], thickness=4, scale=2.3, border=2)

    # Update signal timing based on the interval
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        # Calculate lane densities
        lane_densities = [len(zoneAcounter), len(zoneBcounter), len(zoneCcounter)]
        max_density = max(lane_densities)

        if max_density == 0:
            print("No vehicles detected in any lane. Default signal timing applied.")
            for lane in ['A', 'B', 'C']:
                update_signal_timing(lane, max_green_time / 3)
        else:
            for lane, lane_density in zip(['A', 'B', 'C'], lane_densities):
                signal_time = (lane_density / max_density) * max_green_time
                update_signal_timing(lane, signal_time)

        last_update_time = current_time

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
