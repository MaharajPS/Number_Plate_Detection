import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
from google.cloud import vision
import tempfile
import time
import uuid

# Initialize Google Cloud Vision Client
client = vision.ImageAnnotatorClient()

def lastfour(n):
    count = 0
    i = len(n) - 4 
    while i < len(n):
        c = n[i]
        if c >= '0' and c <= '9':
            count += 1
        else:
            return False
        i += 1
    return True
    

def threendfour(n):
    if len(n) > 4:
        c1 = n[2]
        c2 = n[3]
        if c1 >= '0' and c1 <= '9' and c2 >= '0' and c2 <= '9':
            return True
    return False

def firsttwo(n):
    if len(n) >= 2:
        c1 = n[0]
        c2 = n[1]
        if c1 >= 'A' and c1 <= 'Z' and c2 >= 'A' and c2 <= 'Z':
            return True
    return False
def new(n):
    if len(n)==8:
        return True
    elif len(n)==9:
        c1=n[4]
        if c1 >= 'A' and c1 <= 'Z':
            return True
    elif len(n)==10:
        c1=n[4]
        c2=n[5]
        if c1 >= 'A' and c1 <= 'Z' and c2 >= 'A' and c2 <= 'Z':
            return True
    return False

def cleaning(n):
    result = ""
    for c in n:
        if (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9'):
            result += c
    return result


def imagetostring_from_array(image_array):
    success, encoded_image = cv2.imencode('.jpg', image_array)
    if not success:
        return ""
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        number_plate = cleaning(texts[0].description.strip().replace("\n", ""))
        if "IND" in number_plate:
            number_plate = number_plate.replace("IND", "")
        if 8 <= len(number_plate) <= 10:
            if firsttwo(number_plate) and threendfour(number_plate) and lastfour(number_plate) and new(number_plate):
                return number_plate
    return ""

# Load YOLO Model
model = YOLO("license_plate_detector.pt")

# CSV file
csv_file = 'live_detected_number_plates.csv'
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Timestamp", "Detected Number Plate"]).to_csv(csv_file, index=False)

last_saved = {}

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🔍 Indian Number Plate Detection with OCR")

video_input = st.file_uploader("📹 Upload a Video File", type=['mp4', 'avi', 'mov'])
start_detection = st.button("🚀 Start Detection")

if start_detection and video_input:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_input.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    plate_list = st.empty()
    download_csv = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_plates = []

        if frame_count % 1 == 0:
            results = model(frame)

            for r in results:
                for box in r.boxes:
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
                            detected_plates.append(number_plate)

            # Show frame once per batch
            stframe.image(frame, channels="BGR", caption="Live Detection", use_container_width=False)

        if detected_plates:
            st.success(f"Detected Plate : {detected_plates}")

        if frame_count % 10 == 0:
            df = pd.read_csv(csv_file)
            download_csv.dataframe(df, use_container_width=True)
            download_csv.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                file_name="detected_number_plates.csv",
                key=f"download-{uuid.uuid4()}"
            )

        frame_count += 1
        time.sleep(0.05)

    cap.release()
    st.success("✅ Video processing complete.")
