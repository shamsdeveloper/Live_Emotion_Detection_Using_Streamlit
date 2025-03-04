import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="Advanced Emotion Detection", layout="wide")

@st.cache_resource()
def load_emotion_model():
    try:
        model = load_model("facialemotionmodel.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model

model = load_emotion_model()
detector = MTCNN()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

github_links = {
    "MTCNN": "https://github.com/ipazc/mtcnn",
    "CNN Emotion Model": "https://github.com/oarriaga/face_classification",
    "Multi-Modal Emotion Analysis": "https://github.com/alphacep/vosk-api"
}

st.title("Real-Time Advanced Emotion Detection")
st.markdown("Using CNN & MTCNN for precise emotion recognition and analysis.")

# Emotion data storage
emotion_data = {"Time": [], "Emotion": [], "Confidence": []}

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

run = st.checkbox("Start Webcam", value=st.session_state.camera_active)
stop = st.button("Close Camera")

FRAME_WINDOW = st.empty()
camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_filename = "emotion_detection_output.avi"
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (1280, 720))

sampled_images_path = "saved_frames"
os.makedirs(sampled_images_path, exist_ok=True)
frame_count = 0

feedback_messages = {
    "Happy": "😊 Keep smiling! Happiness is contagious!",
    "Sad": "😢 Consider listening to uplifting music or talking to a friend.",
    "Angry": "😠 Take a deep breath. Try some relaxation techniques.",
    "Fear": "😨 You are safe. Try grounding exercises to stay calm.",
    "Surprise": "😲 Something unexpected! Hope it’s a good surprise!",
    "Neutral": "🙂 Stay relaxed and keep going!"
}

if run and model:
    st.session_state.camera_active = True
    while True:
        ret, frame = camera.read()
        if not ret:
            st.warning("Could not access the webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 720), (50, 50, 50), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=[0, -1])

            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]
            confidence = round(max(prediction) * 100, 2)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            emotion_data["Time"].append(timestamp)
            emotion_data["Emotion"].append(label)
            emotion_data["Confidence"].append(confidence)
            
            cv2.putText(frame, f"Emotion: {label}", (20, 50 + i*80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence}%", (20, 80 + i*80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {timestamp}", (20, 110 + i*80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            st.sidebar.markdown(f"**Emotion Detected:** {label}")
            st.sidebar.write(feedback_messages.get(label, "Stay Positive!"))

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out.write(frame)

        frame_count += 1
        if frame_count % 50 == 0:
            img_name = os.path.join(sampled_images_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(img_name, frame)
        
        if stop:
            break
    
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    st.success("Camera closed. Video saved successfully!")
    
    if os.path.exists(video_filename):
        st.video(video_filename)
    else:
        st.error("Video saving failed. Check permissions or file path.")
    
    st.session_state.camera_active = False

st.subheader("Emotion Trends Over Time")
if len(emotion_data["Time"]) > 0:
    df = pd.DataFrame(emotion_data)
    df["Count"] = 1
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby(["Time", "Emotion"]).count().unstack(fill_value=0).stack()["Count"].unstack().plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    trend_chart_path = "emotion_trends.png"
    plt.savefig(trend_chart_path)
    st.image(trend_chart_path, caption="Emotion Trends Over Time")
else:
    st.write("Emotion data will appear here when the webcam is running.")

st.sidebar.subheader("Project References")
for name, link in github_links.items():
    st.sidebar.markdown(f"[{name}]({link})")

st.sidebar.subheader("Potential Use Cases")
st.sidebar.write("✅ Mental Health Monitoring")
st.sidebar.write("✅ Customer Emotion Analysis")
st.sidebar.write("✅ Education & Student Engagement")
st.sidebar.write("✅ Human-Computer Interaction")
