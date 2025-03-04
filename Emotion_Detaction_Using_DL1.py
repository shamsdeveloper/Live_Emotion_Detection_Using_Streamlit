import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model

# Set TensorFlow environment variable to suppress oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="Elderly Care Emotion Monitoring", layout="wide")

@st.cache_resource()
def load_emotion_model():
    try:
        model = load_model("facialemotionmodel.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model

model = load_emotion_model()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_data = {"Time": [], "Emotion": []}

st.title("Real-Time Emotion Detection for Elderly Care")
st.markdown("This app helps caregivers monitor the emotional well-being of elderly individuals remotely.")

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

run = st.checkbox("Start Webcam", value=st.session_state.camera_active)
stop = st.button("Close Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

# Video writer setup
video_filename = "emotion_video.avi"
frame_size = (1280, 720)
video_writer = None

if run and model:
    st.session_state.camera_active = True

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 10, frame_size)
    
    frame_count = 0
    
    while True:
        ret, frame = camera.read()
        if not ret:
            st.warning("Could not access the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        st.markdown(f"### Number of Faces Detected: {len(faces)}")
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=[0, -1])

            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            y_offset = y - 100 if y > 100 else y + h + 20
            for j, emotion in enumerate(emotion_labels):
                cv2.putText(
                    frame,
                    f"{emotion}: {prediction[j]:.2f}",
                    (x, y_offset + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            emotion_data["Time"].append(datetime.now().strftime("%H:%M:%S"))
            emotion_data["Emotion"].append(label)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_writer.write(frame)
        
        # Save video snapshots every 50 frames
        if frame_count % 50 == 0:
            snapshot_filename = f"snapshot_{frame_count}.jpg"
            cv2.imwrite(snapshot_filename, frame)
            st.image(snapshot_filename, caption=f"Snapshot at frame {frame_count}")
        
        frame_count += 1
        
        if stop:
            break

    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()
    st.success("Camera closed successfully! Video saved.")

else:
    camera.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

# Ensure the saved video is accessible
if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
    with open(video_filename, "rb") as file:
        video_bytes = file.read()
        st.download_button("Download Recorded Video", video_bytes, file_name="emotion_video.avi", mime="video/x-msvideo")
else:
    st.warning("Recorded video could not be saved properly. Please try again.")

st.subheader("Emotion Trends Over Time")
if len(emotion_data["Time"]) > 0:
    df = pd.DataFrame(emotion_data)
    fig, ax = plt.subplots()
    df["Count"] = 1
    emotion_trends = df.groupby(["Time", "Emotion"]).count().unstack(fill_value=0).stack()
    emotion_trends["Count"].unstack().plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    trend_chart_path = "emotion_trends.png"
    plt.savefig(trend_chart_path)
    st.image(trend_chart_path, caption="Emotion Trends Over Time")
else:
    st.write("Emotion data will appear here when the webcam is running.")

# Loss and Accuracy Graphs
st.subheader("Model Loss and Accuracy")
if os.path.exists("model_history.npy"):
    history = np.load("model_history.npy", allow_pickle=True).item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['loss'], label='Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['accuracy'], label='Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    st.pyplot(fig)
else:
    st.write("Model training history not found. Ensure 'model_history.npy' exists.")

st.markdown("This initiative aims to enhance the quality of life for elderly individuals by offering continuous emotional support and enabling caregivers to respond proactively.")
