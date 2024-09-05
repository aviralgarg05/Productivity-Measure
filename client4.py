import cv2
import numpy as np
import asyncio
import websockets
import base64
from deepface import DeepFace
from datetime import datetime, timedelta
from collections import Counter
import time
import csv
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Define save_to_csv function
def save_to_csv(data, filename):
    """Save collected data to a CSV file."""
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Initialize camera with index 2
camera_index = 2
cap = cv2.VideoCapture(camera_index)

# Initialize background removal
segmentor = SelfiSegmentation()

# Initialize variables
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
log_file = 'face_tracking_log.csv'
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
save_to_csv(['Face ID', 'Event', 'Time', 'Duration', 'Cumulative Emotion', 'Productivity Score'], log_file)

face_trackers = {}
face_times = {}
face_emotions = {}
face_last_update = {}
face_id_count = 0
last_detection_time = time.time()
detection_interval = 1  # Interval for face detection

def create_tracker():
    try:
        return cv2.legacy.TrackerKCF_create()
    except Exception as e:
        print(f"Error creating tracker: {e}")
        return None

def rename_emotion(emotion):
    emotion_mapping = {
        'sad': 'stressed',
        'fear': 'tensed'
    }
    return emotion_mapping.get(emotion, emotion)

def analyze_face(face_roi):
    try:
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = rename_emotion(result[0]['dominant_emotion'])
        return emotion
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None

# Updated function to calculate productivity percentage
def calculate_productivity_percentage(emotion_list):
    # Define productive emotions
    productive_emotions = ['happy', 'neutral', 'surprise']

    # Count productive emotions
    productive_count = sum(emotion in productive_emotions for emotion in emotion_list)
    
    # Calculate the percentage of time spent being productive
    total_emotions = len(emotion_list)
    if total_emotions == 0:
        return 0.0  # Avoid division by zero
    
    productivity_percentage = (productive_count / total_emotions) * 100
    return productivity_percentage

# Updated log_productivity function
def log_productivity(face_id, cumulative_emotion, emotion_list, duration):
    # Calculate productivity percentage
    productivity_percentage = calculate_productivity_percentage(emotion_list)

    # Log the result into the CSV
    save_to_csv([face_id, 'productivity', cumulative_emotion, str(duration), f"{productivity_percentage:.2f}%"], log_file)
    
    # Print the result to the console
    print(f"Face ID {face_id} had a productivity percentage of: {productivity_percentage:.2f}%")

def log_time_out(face_id):
    if face_times[face_id]['time_out'] is None:
        face_times[face_id]['time_out'] = get_current_time()
        duration = face_times[face_id]['time_out'] - face_times[face_id]['time_in']
        cumulative_emotion = Counter(face_emotions[face_id]).most_common(1)[0][0]
        save_to_csv([face_id, 'time-out', face_times[face_id]['time_out'], str(duration), cumulative_emotion], log_file)
        log_productivity(face_id, cumulative_emotion, face_emotions[face_id], duration)
        print(f"Face ID {face_id} time-out at: {face_times[face_id]['time_out']}")
        print(f"Face ID {face_id} was present for: {duration}")
        print(f"Face ID {face_id} cumulative emotion: {cumulative_emotion}")

def get_current_time():
    return datetime.now()

def enhance_edges(foreground, background):
    """Enhance the edges of the foreground subject and blend smoothly with background."""
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    bilateral_filtered = cv2.bilateralFilter(gray_foreground, 9, 75, 75)
    edges = cv2.Canny(bilateral_filtered, 50, 150)
    edge_mask = np.zeros_like(foreground, dtype=np.uint8)
    edge_mask[edges != 0] = [0, 0, 0]
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edge_mask = np.zeros_like(foreground, dtype=np.uint8)
    edge_mask[dilated_edges != 0] = [0, 0, 0]
    edge_mask = cv2.GaussianBlur(edge_mask, (15, 15), 0)
    foreground = cv2.addWeighted(foreground, 1.0, edge_mask, 0.4, 0)
    alpha = 0.5
    blended_image = cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)
    return blended_image

def correct_green_tint(frame):
    """Correct greenish tint in the frame."""
    frame_float = frame.astype(np.float32)
    b_mean = np.mean(frame_float[:, :, 0])
    g_mean = np.mean(frame_float[:, :, 1])
    r_mean = np.mean(frame_float[:, :, 2])
    green_correction = (g_mean - b_mean) * 0.5
    frame_float[:, :, 1] -= green_correction
    frame_float = np.clip(frame_float, 0, 255)
    frame_corrected = frame_float.astype(np.uint8)
    return frame_corrected

async def send_frame(websocket, frame):
    try:
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        alpha_channel = np.ones((frame_bgra.shape[0], frame_bgra.shape[1]), dtype=np.uint8) * 0
        frame_bgra[:, :, 3] = alpha_channel
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        _, buffer = cv2.imencode('.png', frame_bgra, encode_param)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(frame_data)
    except Exception as e:
        print(f"Error sending frame: {e}")

async def websocket_loop():
    global last_detection_time, face_id_count
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from camera.")
                continue

            # Display the original feed before processing
            cv2.imshow("Original Feed", frame)

            # Background removal
            background_image = cv2.imread("bg_image.jpeg")
            if background_image is None:
                print("Error: Background image not loaded.")
                continue

            background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
            segmentated_img = segmentor.removeBG(frame, background_image)
            frame = segmentated_img

            # Enhance edges of the foreground
            frame = enhance_edges(frame, background_image)

            # Correct greenish tint
            frame = correct_green_tint(frame)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_ids_to_remove = []
            for face_id, tracker in face_trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    face_roi = frame[y:y + h, x:x + w]
                    emotion = analyze_face(face_roi)
                    if emotion:
                        face_emotions[face_id].append(emotion)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"ID {face_id} {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        face_last_update[face_id] = get_current_time()
                else:
                    log_time_out(face_id)
                    face_ids_to_remove.append(face_id)

            for face_id in face_ids_to_remove:
                face_trackers.pop(face_id, None)
                face_times.pop(face_id, None)
                face_emotions.pop(face_id, None)
                face_last_update.pop(face_id, None)

            current_time = time.time()
            if current_time - last_detection_time > detection_interval:
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
                last_detection_time = current_time
                
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]
                    emotion = analyze_face(face_roi)
                    if emotion:
                        matched_face_id = None
                        for face_id, tracker in face_trackers.items():
                            success, bbox = tracker.update(frame)
                            if success:
                                matched_face_id = face_id
                                break

                        if matched_face_id is None:
                            face_id_count += 1
                            face_id = face_id_count
                            tracker = create_tracker()
                            if tracker is not None:
                                tracker.init(frame, (x, y, w, h))
                                face_trackers[face_id] = tracker
                                face_times[face_id] = {'time_in': get_current_time(), 'time_out': None}
                                face_emotions[face_id] = [emotion]
                                face_last_update[face_id] = get_current_time()
                                save_to_csv([face_id, 'time-in', face_times[face_id]['time_in']], log_file)

            # Send the frame via WebSocket
            await send_frame(websocket, frame)

            # Calculate and display FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Processed Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(websocket_loop())

cap.release()
cv2.destroyAllWindows()
