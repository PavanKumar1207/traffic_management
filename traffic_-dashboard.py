import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *
import time
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import base64

# Initialize YOLO and Tracker
video_path = "vid_one.mp4"
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

# Load traffic light images
traffic_light_images = {
    "red": "images/red_lightt.png",
    "green": "images/green_light.png",
    "orange": "images/orange_light.png"
}
for color, path in traffic_light_images.items():
    try:
        with open(path, "rb") as file:
            print(f"{color} light image found and accessible.")
    except FileNotFoundError:
        print(f"{color} light image not found at {path}.")



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

traffic_light_images_encoded = {
    color: encode_image(path) for color, path in traffic_light_images.items()
}

# Dash App
app = dash.Dash(__name__)
app.title = "Traffic Monitoring Dashboard"

app.layout = html.Div([
    html.H1("Real-Time Traffic Monitoring Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.H3("Signal Lights"),
            html.Div(id="lane-A-signal", style={"textAlign": "center"}),
            html.Div(id="lane-B-signal", style={"textAlign": "center"}),
            html.Div(id="lane-C-signal", style={"textAlign": "center"}),
        ], style={"width": "25%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            html.H3("Live Video Feed"),
            html.Img(id="video-frame", style={"width": "70%"})
        ], style={"width": "70%", "display": "inline-block"}),
    ], style={"display": "flex", "justifyContent": "space-between"}),

    dcc.Interval(id="video-update", interval=1000, n_intervals=0)
])

@app.callback([
    Output("video-frame", "src"),
    Output("lane-A-signal", "children"),
    Output("lane-B-signal", "children"),
    Output("lane-C-signal", "children")
], [Input("video-update", "n_intervals")])
def update_dashboard(n):
    global last_update_time, lane_signals, zoneAcounter, zoneBcounter, zoneCcounter

    ret, frame = cap.read()
    if not ret:
        return "", "Video Finished", "", ""

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
    if current_time - last_update_time >= 10:
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

    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')

    return (
        f"data:image/jpeg;base64,{frame_data}",
        html.Img(src=f"data:image/png;base64,{traffic_light_images_encoded[lane_signals['A']]}", style={"width": "100px"}),
        html.Img(src=f"data:image/png;base64,{traffic_light_images_encoded[lane_signals['B']]}", style={"width": "100px"}),
        html.Img(src=f"data:image/png;base64,{traffic_light_images_encoded[lane_signals['C']]}", style={"width": "100px"})
    )

if __name__ == "__main__":
    cap = cv2.VideoCapture(video_path)
    app.run_server(debug=True)
