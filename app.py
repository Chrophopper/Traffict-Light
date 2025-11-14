from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
from ultralytics import YOLO
import threading
import time
import datetime
import yt_dlp
import re
import os
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

DATA_FILE = 'data.pkl'

# In-memory storage for roads and lanes (will be loaded/saved from/to DATA_FILE)
roads_data = []
next_road_id = 1
next_lane_id = 1

# In-memory storage for detection results and history (these are dynamic, not persistent)
detection_results = {} # {road_id: {lane_id: {vehicle_counts: {}, speed: 0, density: "low"}}}
detection_history = [] # List of detection snapshots

# YOLO Models (assuming they are in the 'Model' folder)
model_n = YOLO('Model/yolo11n.pt')
model_s = YOLO('Model/yolo11s.pt')

# Define a list of relevant vehicle types to filter the YOLO model's classes
RELEVANT_VEHICLE_TYPES = [
    'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'person', 'pedicab' # Added 'person' and 'pedicab' based on previous default
]

# Dynamically get vehicle classes from the model and filter them
if hasattr(model_n, 'names') and model_n.names:
    all_model_classes = {int(k): v for k, v in model_n.names.items()}
    VEHICLE_CLASSES = {
        k: v for k, v in all_model_classes.items()
        if v in RELEVANT_VEHICLE_TYPES
    }
    print(f"Dynamically loaded and filtered VEHICLE_CLASSES from model: {VEHICLE_CLASSES}")
else:
    # Fallback to a predefined list if model names are not available or for specific needs
    VEHICLE_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'pedicab',
        5: 'bus',
        6: 'person',
        7: 'truck'
    }
    print(f"Using default VEHICLE_CLASSES (model names not available or filtered): {VEHICLE_CLASSES}")

# Traffic density thresholds (example, adjust as needed)
DENSITY_THRESHOLDS = {
    'low': {'count': 0, 'speed': 30},
    'medium': {'count': 5, 'speed': 20},
    'high': {'count': 10, 'speed': 10}
}

# Traffic light control (example)
traffic_light_status = {} # {road_id: {lane_id: "red" or "green"}}
traffic_light_duration = 30 # seconds for green light

# Function to extract direct video URL from YouTube link
def get_youtube_direct_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'noplaylist': True,
        'quiet': True,
        'simulate': True,
        'force_url': True,
        'retries': 3,
    }
    try:
        print(f"yt_dlp: Attempting to extract info for {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if 'url' in info:
                print(f"yt_dlp: Successfully extracted direct URL: {info['url']}")
                return info['url']
            elif 'entries' in info and info['entries']:
                print(f"yt_dlp: Successfully extracted direct URL from entry: {info['entries'][0]['url']}")
                return info['entries'][0]['url']
            else:
                print(f"yt_dlp: No direct URL found in info for {youtube_url}")
    except Exception as e:
        print(f"yt_dlp: Error extracting YouTube URL for {youtube_url}: {e}")
    return None

# Function to process video stream and perform detection
def process_video_stream(road_id, lane_id, data_source):
    final_data_source = data_source
    is_youtube_source = "youtube.com" in data_source or "youtu.be" in data_source

    if is_youtube_source:
        print(f"process_video_stream: Attempting to extract direct URL for YouTube source: {data_source}")
        final_data_source = get_youtube_direct_url(data_source)
        if not final_data_source:
            print(f"process_video_stream: Failed to get direct YouTube URL for {data_source}. Falling back to original.")
            final_data_source = data_source
        else:
            print(f"process_video_stream: Using extracted direct URL: {final_data_source}")
    else:
        print(f"process_video_stream: Using local/direct video source: {data_source}")

    cap = cv2.VideoCapture(final_data_source)
    if not cap.isOpened():
        error_message = f"Error: Could not open video source {final_data_source}. Please check the URL/path and ensure OpenCV can access it. This might be due to an invalid URL, network issues, or missing codecs."
        print(error_message)
        if road_id not in detection_results:
            detection_results[road_id] = {}
        detection_results[road_id][lane_id] = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'vehicle_counts': {v_type: 0 for v_type in VEHICLE_CLASSES.values()},
            'total_vehicles': 0,
            'speed': 0,
            'density': 'error',
            'frame': None,
            'error_message': error_message
        }
        return

    print(f"process_video_stream: Successfully opened video source {final_data_source}")

    frame_skip = 5 # Process every 5th frame
    frame_count = 0
    youtube_url_refresh_interval = 300 # Refresh YouTube URL every 5 minutes (300 seconds)
    last_youtube_url_refresh = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"process_video_stream: End of video stream or frame read error for {data_source}. Attempting to loop or re-fetch URL.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Try to loop video
            ret, frame = cap.read()
            if not ret:
                # If looping fails, and it's a YouTube source, try to re-fetch the URL
                if is_youtube_source and (time.time() - last_youtube_url_refresh > youtube_url_refresh_interval):
                    print(f"process_video_stream: Re-fetching YouTube URL for {data_source} due to stream end/error.")
                    new_direct_url = get_youtube_direct_url(data_source)
                    if new_direct_url:
                        final_data_source = new_direct_url
                        cap.release()
                        cap = cv2.VideoCapture(final_data_source)
                        if not cap.isOpened():
                            error_message = f"Error: Could not re-open video source {final_data_source} after URL refresh."
                            print(error_message)
                            detection_results[road_id][lane_id]['error_message'] = error_message
                            time.sleep(5) # Wait before retrying
                            continue
                        else:
                            print(f"process_video_stream: Successfully re-opened video source with new URL: {final_data_source}")
                            last_youtube_url_refresh = time.time()
                            continue # Try reading frame again with new cap
                    else:
                        print(f"process_video_stream: Failed to re-fetch YouTube URL for {data_source}. Exiting stream processing.")
                        detection_results[road_id][lane_id]['error_message'] = f"Stream ended or failed to re-fetch URL for {data_source}."
                        break
                else:
                    print(f"process_video_stream: Failed to read frame after loop for {data_source}. Exiting stream processing.")
                    detection_results[road_id][lane_id]['error_message'] = f"Stream ended or failed to loop for {data_source}."
                    break
            continue

        # Resize frame immediately after reading to prevent memory issues with large frames
        # Target resolution: 640x384 (or adjust as needed)
        if frame is None:
            print(f"process_video_stream: Frame is None after cap.read() for {data_source}. Skipping frame.")
            time.sleep(0.1) # Prevent busy-waiting
            continue
        
        processed_frame = cv2.resize(frame, (640, 384))

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        try:
            results = model_n.predict(processed_frame, conf=0.5, iou=0.7, classes=list(VEHICLE_CLASSES.keys()))
        except Exception as e:
            print(f"process_video_stream: Error during YOLO prediction for {data_source}: {e}")
            annotated_frame = processed_frame.copy()
            results = []

        current_vehicle_counts = {v_type: 0 for v_type in VEHICLE_CLASSES.values()}
        current_speeds = []

        annotated_frame = frame.copy()
        for r in results:
            annotated_frame = r.plot()

            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_type = VEHICLE_CLASSES[cls]
                    current_vehicle_counts[vehicle_type] += 1
                    current_speeds.append(25)

        total_vehicles = sum(current_vehicle_counts.values())
        avg_speed = sum(current_speeds) / len(current_speeds) if current_speeds else 0

        density_status = "low"
        if total_vehicles >= DENSITY_THRESHOLDS['high']['count'] and avg_speed <= DENSITY_THRESHOLDS['high']['speed']:
            density_status = "high"
        elif total_vehicles >= DENSITY_THRESHOLDS['medium']['count'] and avg_speed <= DENSITY_THRESHOLDS['medium']['speed']:
            density_status = "medium"

        if road_id not in detection_results:
            detection_results[road_id] = {}
        detection_results[road_id][lane_id] = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'vehicle_counts': current_vehicle_counts,
            'total_vehicles': total_vehicles,
            'speed': round(avg_speed, 2),
            'density': density_status,
            'frame': annotated_frame,
            'error_message': None
        }

    cap.release()
    print(f"process_video_stream: Released video capture for {data_source}")

# Generator function to stream video frames
def generate_frames(road_id, lane_id):
    while True:
        if road_id in detection_results and lane_id in detection_results[road_id]:
            frame = detection_results[road_id][lane_id].get('frame')
            error_message = detection_results[road_id][lane_id].get('error_message')

            if error_message:
                blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                text_lines = ["Error: Video Feed Failed", "Check terminal for details."]
                if error_message and len(error_message) > 40:
                    text_lines.append(error_message[:35] + "...")
                elif error_message:
                    text_lines.append(error_message)

                y_offset = 80
                for line in text_lines:
                    cv2.putText(blank_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 30
                
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            elif frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        print(f"generate_frames: Failed to encode frame for road {road_id}, lane {lane_id}")
                except Exception as e:
                    print(f"generate_frames: Error during frame encoding for road {road_id}, lane {lane_id}: {e}")
            else:
                blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                text = "Loading Video Feed..."
                cv2.putText(blank_frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

# Thread management for detection
detection_threads = {} # { (road_id, lane_id): thread_object }

def start_detection_for_lane(road_id, lane_id, data_source):
    thread_key = (road_id, lane_id)
    if thread_key in detection_threads and detection_threads[thread_key].is_alive():
        print(f"Stopping existing detection thread for lane {lane_id} on road {road_id}.")
        del detection_threads[thread_key]

    thread = threading.Thread(target=process_video_stream, args=(road_id, lane_id, data_source))
    thread.daemon = True
    thread.start()
    detection_threads[thread_key] = thread
    print(f"Started detection for lane {lane_id} on road {road_id} from {data_source}")

# Traffic light logic
def update_traffic_lights():
    while True:
        for road_id, lanes_data in detection_results.items():
            if not lanes_data:
                continue

            most_congested_lane_id = None
            max_density_score = -1

            for lane_id, data in lanes_data.items():
                density_score = 0
                if data['density'] == 'high':
                    density_score = 3
                elif data['density'] == 'medium':
                    density_score = 2
                elif data['density'] == 'low':
                    density_score = 1

                density_score += data['total_vehicles'] * 0.1

                if density_score > max_density_score:
                    max_density_score = density_score
                    most_congested_lane_id = lane_id

            if road_id not in traffic_light_status:
                traffic_light_status[road_id] = {}

            for lane_id in lanes_data.keys():
                if lane_id == most_congested_lane_id:
                    traffic_light_status[road_id][lane_id] = "green"
                else:
                    traffic_light_status[road_id][lane_id] = "red"

        snapshot = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': detection_results.copy(),
            'traffic_lights': traffic_light_status.copy()
        }
        detection_history.append(snapshot)
        if len(detection_history) > 100:
            detection_history.pop(0)

        time.sleep(traffic_light_duration)

# Start traffic light update thread
traffic_light_thread = threading.Thread(target=update_traffic_lights)
traffic_light_thread.daemon = True
traffic_light_thread.start()

def load_data():
    global roads_data, next_road_id, next_lane_id
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
            roads_data.extend(data.get('roads_data', []))
            next_road_id = data.get('next_road_id', 1)
            next_lane_id = data.get('next_lane_id', 1)
        print(f"Data loaded from {DATA_FILE}: roads_data={roads_data}, next_road_id={next_road_id}, next_lane_id={next_lane_id}")
    else:
        print(f"No data file found at {DATA_FILE}. Starting with empty data.")

def save_data():
    data = {
        'roads_data': roads_data,
        'next_road_id': next_road_id,
        'next_lane_id': next_lane_id
    }
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {DATA_FILE}: roads_data={roads_data}, next_road_id={next_road_id}, next_lane_id={next_lane_id}")

# Load data when the application starts
load_data()

# Start detection threads for all existing lanes
for road in roads_data:
    for lane in road['lanes']:
        start_detection_for_lane(road['id'], lane['id'], lane['data_source'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/roads')
def roads_list():
    return render_template('roads.html', roads=roads_data)

@app.route('/roads/add', methods=['GET', 'POST'])
def add_road():
    global next_road_id
    if request.method == 'POST':
        road_name = request.form['road_name']
        road_description = request.form['road_description']
        new_road = {
            'id': next_road_id,
            'name': road_name,
            'description': road_description,
            'lanes': []
        }
        roads_data.append(new_road)
        next_road_id += 1
        save_data()
        flash(f"Persimpangan '{road_name}' berhasil ditambahkan.", 'success')
        return redirect(url_for('roads_list'))
    return render_template('add_road.html')

@app.route('/roads/edit/<int:road_id>', methods=['GET', 'POST'])
def edit_road(road_id):
    road = next((r for r in roads_data if r['id'] == road_id), None)
    if road is None:
        flash(f"Persimpangan dengan ID {road_id} tidak ditemukan.", 'error')
        return redirect(url_for('roads_list'))

    if request.method == 'POST':
        road['name'] = request.form['road_name']
        road['description'] = request.form['road_description']
        save_data()
        flash(f"Persimpangan '{road['name']}' berhasil diperbarui.", 'success')
        return redirect(url_for('roads_list'))
    return render_template('edit_road.html', road=road)

@app.route('/roads/delete/<int:road_id>', methods=['POST'])
def delete_road(road_id):
    global roads_data
    initial_len = len(roads_data)
    roads_data = [r for r in roads_data if r['id'] != road_id]
    if len(roads_data) < initial_len:
        save_data()
        flash(f"Persimpangan dengan ID {road_id} berhasil dihapus.", 'success')
    else:
        flash(f"Persimpangan dengan ID {road_id} tidak ditemukan.", 'error')
    return redirect(url_for('roads_list'))

@app.route('/roads/<int:road_id>/lanes/add', methods=['GET', 'POST'])
def add_lane(road_id):
    global next_lane_id
    road = next((r for r in roads_data if r['id'] == road_id), None)
    if road is None:
        flash(f"Persimpangan dengan ID {road_id} tidak ditemukan.", 'error')
        return redirect(url_for('roads_list'))

    if request.method == 'POST':
        lane_name = request.form['lane_name']
        lane_data_source = request.form['lane_data_source']
        new_lane = {
            'id': next_lane_id,
            'name': lane_name,
            'data_source': lane_data_source
        }
        road['lanes'].append(new_lane)
        next_lane_id += 1
        save_data()
        start_detection_for_lane(road_id, new_lane['id'], lane_data_source)
        flash(f"Jalur '{lane_name}' berhasil ditambahkan ke persimpangan '{road['name']}'.", 'success')
        return redirect(url_for('roads_list'))
    return render_template('add_lane.html', road=road)

@app.route('/roads/<int:road_id>/lanes/edit/<int:lane_id>', methods=['GET', 'POST'])
def edit_lane(road_id, lane_id):
    road = next((r for r in roads_data if r['id'] == road_id), None)
    if road is None:
        flash(f"Persimpangan dengan ID {road_id} tidak ditemukan.", 'error')
        return redirect(url_for('roads_list'))
    lane = next((l for l in road['lanes'] if l['id'] == lane_id), None)
    if lane is None:
        flash(f"Jalur dengan ID {lane_id} tidak ditemukan di persimpangan '{road['name']}'.", 'error')
        return redirect(url_for('roads_list'))

    if request.method == 'POST':
        lane['name'] = request.form['lane_name']
        lane['data_source'] = request.form['lane_data_source']
        start_detection_for_lane(road_id, lane_id, lane['data_source'])
        save_data()
        flash(f"Jalur '{lane['name']}' berhasil diperbarui.", 'success')
        return redirect(url_for('roads_list'))
    return render_template('edit_lane.html', road=road, lane=lane)

@app.route('/roads/<int:road_id>/lanes/delete/<int:lane_id>', methods=['POST'])
def delete_lane(road_id, lane_id):
    road = next((r for r in roads_data if r['id'] == road_id), None)
    if road is None:
        flash(f"Persimpangan dengan ID {road_id} tidak ditemukan.", 'error')
        return redirect(url_for('roads_list'))
    
    initial_len = len(road['lanes'])
    road['lanes'] = [l for l in road['lanes'] if l['id'] != lane_id]
    if len(road['lanes']) < initial_len:
        save_data()
        flash(f"Jalur dengan ID {lane_id} berhasil dihapus dari persimpangan '{road['name']}'.", 'success')
    else:
        flash(f"Jalur dengan ID {lane_id} tidak ditemukan di persimpangan '{road['name']}'.", 'error')

    thread_key = (road_id, lane_id)
    if thread_key in detection_threads:
        del detection_threads[thread_key]
        print(f"Stopped tracking detection thread for lane {lane_id} on road {road_id}")
    return redirect(url_for('roads_list'))

@app.route('/dashboard')
def dashboard():
    dashboard_data = {}
    for road in roads_data:
        road_id = road['id']
        road_name = road.get('name', f'Road {road_id}')
        dashboard_data[road_name] = {
            'road_id': road_id,
            'lanes': [],
            'traffic_lights': traffic_light_status.get(road_id, {})
        }
        for lane in road['lanes']:
            lane_id = lane['id']
            current_detection = detection_results.get(road_id, {}).get(lane_id, {})
            
            lane_info = {
                'lane_id': lane_id,
                'name': lane['name'],
                'data_source': lane['data_source'],
                'vehicle_counts': current_detection.get('vehicle_counts', {v_type: 0 for v_type in VEHICLE_CLASSES.values()}),
                'total_vehicles': current_detection.get('total_vehicles', 0),
                'speed': current_detection.get('speed', 0),
                'density': current_detection.get('density', 'low'),
                'timestamp': current_detection.get('timestamp', 'N/A'),
                'light_status': traffic_light_status.get(road_id, {}).get(lane_id, 'unknown')
            }
            dashboard_data[road_name]['lanes'].append(lane_info)
    return render_template('dashboard.html', dashboard_data=dashboard_data, roads_data=roads_data)

@app.route('/history')
def history():
    return render_template('history.html', history=detection_history, roads_data=roads_data)

@app.route('/simulation')
def simulation():
    return render_template('simulation.html', roads=roads_data, detection_results=detection_results, traffic_light_status=traffic_light_status)

@app.route('/video_feed/<int:road_id>/<int:lane_id>')
def video_feed(road_id, lane_id):
    return Response(generate_frames(road_id, lane_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
