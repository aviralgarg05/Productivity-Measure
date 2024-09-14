import cv2
import numpy as np
import base64
from deepface import DeepFace
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import mediapipe as mp
import uvicorn
import threading
import time
from collections import defaultdict
from datetime import datetime
import csv

app = FastAPI()

# Configure logging for both general logs and base64 string logs
logging.basicConfig(level=logging.INFO, filename="app_logs.log", filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Separate log file for base64 strings
base64_log_file = "base64_logs.log"

# Initialize background removal with mediapipe's SelfieSegmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize variables
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Global dictionary to track detected faces
face_data = defaultdict(lambda: {"time_in": None, "time_out": None, "emotions": [], "base64_image": None})
next_face_id = 0

# Initialize the webcam capture (global object for sharing across requests)
camera = cv2.VideoCapture(0)

# Emotion productivity weights (can be adjusted based on your needs)
emotion_weights = {
    "happy": 1.0,
    "neutral": 0.7,
    "surprise": 0.5,
    "sad": -0.5,
    "angry": -1.0,
    "fear": -0.8,
    "disgust": -0.7,
}

# CSV file path for storing face logs
csv_file_path = 'face_data.csv'

# Initialize the CSV file by writing the headers if the file doesn't exist
def init_csv():
    try:
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            file.seek(0)  # Move to the beginning of the file
            if file.read(1):  # Check if the file is not empty (already has header)
                return
            # Write header if CSV is empty
            writer.writerow(['face_id', 'time_in', 'time_out', 'duration', 'emotions', 'productivity'])
        logging.info("CSV file initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the CSV file: {e}")

# Function to insert face data into the CSV file
def insert_face_log(face_id, time_in, time_out, duration, emotions, productivity):
    try:
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([face_id, time_in, time_out, duration, ','.join(emotions), productivity])
        logging.info(f"Inserted face data for Face ID {face_id} into CSV file.")
    except Exception as e:
        logging.error(f"Error inserting face data into CSV: {e}")

# Function to log base64 strings to a separate file
def log_base64(face_id, base64_str):
    with open(base64_log_file, 'a') as f:
        f.write(f"Face ID: {face_id}, Base64: {base64_str}\n")

# Helper function to convert base64 string to OpenCV image
def base64_to_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"Error decoding base64 image: {e}")
        return None

# Function to convert OpenCV image to base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# Analyze face for emotion using DeepFace
def analyze_face(face_roi):
    try:
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotion = result.get('dominant_emotion')
        return emotion
    except Exception as e:
        logging.error(f"Error analyzing face: {e}")
        return None

# Function to calculate productivity based on emotions
def calculate_productivity(emotions):
    if not emotions:
        return 0.0
    weighted_sum = sum(emotion_weights.get(emotion, 0) for emotion in emotions)
    return max(0, min(100, (weighted_sum / len(emotions)) * 100))  # Clamp between 0 and 100

# Function to match or assign a new face ID using DeepFace embeddings
def get_or_assign_face_id(face_roi, face_ids):
    global next_face_id

    # Get face embedding using DeepFace
    try:
        face_embedding = DeepFace.represent(face_roi, enforce_detection=False)[0]['embedding']
    except Exception as e:
        logging.error(f"Error extracting face embedding: {e}")
        return None

    # Try to match with existing face embeddings
    for face_id, data in face_ids.items():
        if 'embedding' in data and np.linalg.norm(np.array(data['embedding']) - np.array(face_embedding)) < 0.6:
            return face_id

    # If no match, assign a new face ID
    face_ids[next_face_id] = {"embedding": face_embedding}
    assigned_face_id = next_face_id
    next_face_id += 1
    return assigned_face_id

# Function to continuously capture frames from the camera
def capture_and_process_video():
    face_ids = {}  # Dictionary to store face embeddings and face IDs
    while True:
        ret, frame = camera.read()  # Read a frame from the webcam
        if not ret:
            logging.error("Failed to capture frame from webcam")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        detected_face_ids = set()  # Track the faces detected in this frame

        # If faces are found, process each face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_id = get_or_assign_face_id(face_roi, face_ids)

            if face_id is None:
                continue

            detected_face_ids.add(face_id)

            # If it's the first time we see this face, log time_in and store base64 image
            if face_data[face_id]["time_in"] is None:
                face_data[face_id]["time_in"] = current_time
                face_data[face_id]["base64_image"] = image_to_base64(face_roi)
                log_base64(face_id, face_data[face_id]["base64_image"])  # Log the base64 string

            # Analyze face for emotion
            emotion = analyze_face(face_roi)
            if emotion:
                face_data[face_id]["emotions"].append(emotion)
                logging.info(f"Face ID {face_id} detected emotion: {emotion}")

        # Handle faces that are no longer in the frame
        for face_id in list(face_data.keys()):
            if face_id not in detected_face_ids and face_data[face_id]["time_out"] is None:
                face_data[face_id]["time_out"] = current_time
                time_in = datetime.strptime(face_data[face_id]["time_in"], '%Y-%m-%d %H:%M:%S')
                time_out = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
                duration = (time_out - time_in).total_seconds()  # Duration in seconds

                cumulative_emotions = face_data[face_id]["emotions"]
                productivity = calculate_productivity(cumulative_emotions)

                # Log the face data
                logging.info(f"Face ID {face_id} left. Time in: {time_in}, Time out: {time_out}. "
                             f"Duration: {duration} seconds, Cumulative emotions: {cumulative_emotions}, Productivity: {productivity}%")
                
                # Store the data in CSV
                insert_face_log(face_id, face_data[face_id]["time_in"], current_time, duration, cumulative_emotions, productivity)

        # Display the frame with detection (for debugging purposes)
        cv2.imshow('Emotion Detection', frame)

        # Stop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Endpoint to start the video feed and emotion detection
@app.post("/process_frame")
async def process_frame():
    try:
        # Start the video capture and processing in a separate thread
        thread = threading.Thread(target=capture_and_process_video)
        thread.start()

        return {"message": "Video processing started. Check logs for emotion detection results."}
    except Exception as e:
        logging.error(f"Error starting video feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start video feed")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


if __name__ == "__main__":
    # Initialize the CSV file
    init_csv()

    # Run the FastAPI app (or any other main code)
    uvicorn.run(app, host="0.0.0.0", port=8000)
