import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *
import time
import streamlit as st
import folium
from streamlit_folium import st_folium

# Initialize Streamlit components
st.set_page_config(page_title="Traffic Monitoring Dashboard", layout="wide")
st.title("🌐 Real-Time Traffic Monitoring Dashboard")
st.sidebar.header("🔧 Configuration")

# Sidebar Inputs
video_path = st.sidebar.text_input("Video Path", "vid_one.mp4")
max_green_time = st.sidebar.slider("Max Green Time (seconds)", 30, 180, 120)
update_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 10)
location_input = st.sidebar.text_input("Enter Location (Latitude, Longitude)", "37.7749, -122.4194")

# Display the map
lat, lon = map(float, location_input.split(","))
map_object = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon], popup="Traffic Location", tooltip="Click for details").add_to(map_object)
st.sidebar.markdown("### 📍 Location on Map")
st_folium(map_object, width=350)

# Load traffic light images
traffic_light_images = {
    "red": "D:/computer vision/Vehicle-Counting-in-Lanes/red_light.png",
    "green": "D:/computer vision/Vehicle-Counting-in-Lanes/green_light.png",
    "orange": "D:/computer vision/Vehicle-Counting-in-Lanes/orange_light.png"
}
st.image("D:/computer vision/Vehicle-Counting-in-Lanes/red_light.png", caption="Red Light")


# Initialize YOLO and Tracker
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')
tracker = Sort()

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

road_zoneA = np.array([[308, 789], [711, 807], [694, 492], [415, 492], [309, 790]], np.int32)
road_zoneB = np.array([[727, 797], [1123, 812], [1001, 516], [741, 525], [730, 795]], np.int32)
road_zoneC = np.array([[1116, 701], [1533, 581], [1236, 367], [1009, 442], [1122, 698]], np.int32)

zoneA_Line = np.array([road_zoneA[0], road_zoneA[1]]).reshape(-1)
zoneB_Line = np.array([road_zoneB[0], road_zoneB[1]]).reshape(-1)
zoneC_Line = np.array([road_zoneC[0], road_zoneC[1]]).reshape(-1)

zoneAcounter, zoneBcounter, zoneCcounter = [], [], []
last_update_time = time.time()

# Initialize signal states
lane_signals = {'A': 'red', 'B': 'red', 'C': 'red'}

def display_signal_light(lane, color):
    st.markdown(f"### {lane}")
    st.image(traffic_light_images[color], width=100)

# Main Dashboard Layout
frame_placeholder = st.empty()
signal_column = st.sidebar.container()
st.sidebar.markdown("### 🚦 Signal Timing Adjustments")
st.markdown("<hr>", unsafe_allow_html=True)

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Video playback finished.")
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
            if class_detect in ['car', 'truck', 'bus'] and conf > 60:
                detections = np.array([x1, y1, x2, y2, conf])
                current_detections = np.vstack([current_detections, detections])

    track_results = tracker.update(current_detections)
    for result in track_results:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2 - 40

        if zoneA_Line[0] < cx < zoneA_Line[2] and zoneA_Line[1] - 20 < cy < zoneA_Line[1] + 20:
            if id not in zoneAcounter:
                zoneAcounter.append(id)

        if zoneB_Line[0] < cx < zoneB_Line[2] and zoneB_Line[1] - 20 < cy < zoneB_Line[1] + 20:
            if id not in zoneBcounter:
                zoneBcounter.append(id)

        if zoneC_Line[0] < cx < zoneC_Line[2] and zoneC_Line[1] - 20 < cy < zoneC_Line[1] + 20:
            if id not in zoneCcounter:
                zoneCcounter.append(id)

    # Update signal timing based on the interval
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        lane_densities = [len(zoneAcounter), len(zoneBcounter), len(zoneCcounter)]
        max_density = max(lane_densities)

        for lane, lane_density in zip(['A', 'B', 'C'], lane_densities):
            if max_density == 0:
                lane_signals[lane] = 'red'
            elif lane_density == max_density:
                lane_signals[lane] = 'green'
            else:
                lane_signals[lane] = 'red'

        last_update_time = current_time

    # Update vehicle counts and signals
    with signal_column:
        st.markdown("### 🚦 Signal Lights")
        display_signal_light("LANE A", lane_signals['A'])
        display_signal_light("LANE B", lane_signals['B'])
        display_signal_light("LANE C", lane_signals['C'])

    frame_placeholder.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
