import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
from google.cloud import vision
import tempfile

# --- Validation Functions ---
def lastfour(n):
    i = len(n) - 4
    while i < len(n):
        c = n[i]
        if c < '0' or c > '9':
            return False
        i += 1
    return True

def threendfour(n):
    if len(n) > 4:
        return n[3].isdigit() and n[4].isdigit()
    return False

def firsttwo(n):
    return len(n) >= 2 and 'A' <= n[0] <= 'Z' and 'A' <= n[1] <= 'Z'

def clean_plate(s):
    result = ""
    for c in s:
        if 'A' <= c <= 'Z' or '0' <= c <= '9':
            result += c
    return result

def imagetostring_from_array(image_array):
    client = vision.ImageAnnotatorClient()
    success, encoded_image = cv2.imencode('.jpg', image_array)
    if not success:
        return ""
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        number_plate = texts[0].description.strip().replace("\n", "").replace(" ", "").replace("IND", "")
        number_plate = clean_plate(number_plate)
        if 8 <= len(number_plate) <= 10:
            if firsttwo(number_plate) and threendfour(number_plate) and lastfour(number_plate):
                return number_plate
    return ""

# --- Streamlit App ---
st.title("📹 Number Plate Detection from Video")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Display uploaded video
    st.video(uploaded_video)

    # Load model
    model = YOLO("license_plate_detector.pt")

    # CSV setup
    csv_file = 'live_detected_number_plates.csv'
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=["Timestamp", "Detected Number Plate"]).to_csv(csv_file, index=False)
    last_saved = {}

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    plate_id = 0

    stframe = st.empty()
    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 10 == 0:  # Process every 10th frame
            results = model(frame)
            for r in results:
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_plate = frame[y1:y2, x1:x2]
                    number_plate = imagetostring_from_array(cropped_plate)

                    if number_plate:
                        now = datetime.now()
                        if number_plate not in last_saved or (now - last_saved[number_plate]) > timedelta(minutes=5):
                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            pd.DataFrame([[timestamp, number_plate]], columns=["Timestamp", "Detected Number Plate"]).to_csv(
                                csv_file, mode='a', header=False, index=False)
                            last_saved[number_plate] = now
                            st.success(f"✅ Detected: `{number_plate}`")
                        else:
                            st.warning(f"⚠️ Skipped (within 5 minutes): `{number_plate}`")
                    else:
                        st.error("❌ Plate unreadable.")

                    st.image(cropped_plate, caption="Detected Plate", width=300)
                    plate_id += 1

        frame_count += 1
        processed += 1
        progress.progress(min(processed / total_frames, 1.0))

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        st.subheader("📋 Detected Number Plates")
        st.dataframe(df)
    cap.release()
    st.success("✅ Video processing complete!")
