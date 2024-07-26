import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt
import json
from streamlit_lottie import st_lottie

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.write(f"Error loading cascade classifiers: {e}")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self._initialized = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                label_position = (x, y - 10)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def process_uploaded_image(uploaded_image):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
    emotion_counts = {emotion: 0 for emotion in emotion_labels}

    # Create a copy for visualization
    img_vis = img.copy()

    for (x, y, w, h) in faces:
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            finalout = emotion_labels[maxindex]
            emotion_counts[finalout] += 1

            # Draw rectangle and label on the image
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label_position = (x, y - 10)
            cv2.putText(img_vis, finalout, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img, img_vis, emotion_counts

def plot_emotion_counts(emotion_counts):
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    fig, ax = plt.subplots()
    ax.bar(emotions, counts, color='skyblue')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Counts')
    ax.set_title('Detected Emotions')

    # Add value labels on bars
    for i, count in enumerate(counts):
        ax.text(i, count + 0.1, str(count), ha='center')

    st.pyplot(fig)

def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.write(f"Error loading Lottie file: {e}")
        return None

def main():
    st.set_page_config(page_title="Face Emotion Detection", page_icon="😃", layout="centered")

    st.title("Welcome to Face Emotion Detection")

    # Define pages
    pages = {
        "Home": "home",
        "Live Emotion Detection": "live",
        "Image Emotion Detection": "upload",
        "About": "about"
    }

    choice = st.sidebar.selectbox("Select Options", list(pages.keys()))

    # CSS for homepage
    st.markdown("""
    <style>
    .centered-animation {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .content-box {
        background: linear-gradient(to top left, #1e130c, #9a8478);
        padding: 20px;
        margin: 10px 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        color: #ffffff;
        font-weight: bold;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Homepage
    if choice == "Home":
        st.markdown("""
        <div class="centered-animation">
        """, unsafe_allow_html=True)
        lottie_file = "Animation - 1721976268168.json"  # Update with your local Lottie file path
        lottie_json = load_lottie_file(lottie_file)
        if lottie_json:
            st_lottie(lottie_json, speed=1, width=700, height=300, key="lottie_animation")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="content-box">
        <ol>
            <li>If you want to detect emotions in real time using your webcam, select "Live Face Emotion Detection" from the side menu.</li>
            <li>If you have an image and want to detect emotions in it, select "Upload Image for Emotion Detection" from the side menu.</li>
            <li>Recognizing facial emotions is essential for enhancing communication during online classes, meetings, and virtual interactions.</li>
            <li>This tool can help analyze and improve cognitive engagement by detecting emotions.</li>
            <li>Emotion recognition can provide valuable insights for research purposes during virtual classes or meetings.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        # Add a welcoming pop-up
        st.markdown("""
        <script>
        alert("Welcome to the Face Emotion Detection app! Get ready to analyze and enhance cognitive engagement.");
        </script>
        """, unsafe_allow_html=True)

    # Live Face Emotion Detection
    elif choice == "Live Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader('Get ready with all the emotions you can express.')
        st.write("1. Click Start to open your camera and give permission for prediction.")
        st.write("2. This will predict your emotion.")
        st.write("3. When you're done, click Stop to end.")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # Upload Image for Emotion Detection
    elif choice == "Image Emotion Detection":
        st.header("Upload Image")
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_image is not None:
            img, img_vis, emotion_counts = process_uploaded_image(uploaded_image)

            # Display original and processed images side by side
            st.write("Original Image:")
            st.image(img, caption='Uploaded Image', use_column_width=True)
            st.write("Emotion Detected Image:")
            st.image(img_vis, caption='Image with Detected Emotions', use_column_width=True)

            st.write("Detecting emotions...")
            plot_emotion_counts(emotion_counts)
            st.write("Emotion Counts:", emotion_counts)

    # About
    elif choice == "About":
        st.subheader("About this app")
        st.markdown("""
        <div style="background-color:#36454F;padding:30px; color:white;">
        <h4>This app predicts facial emotion using a Convolutional Neural Network, built with Keras and TensorFlow libraries.
        Face detection is achieved through OpenCV.</h4>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; background: linear-gradient(to bottom right, #ff00cc, #333399); padding: 10px; text-align: center;">
        <p style="
            margin: 0; 
            font-size: 1em; 
            color: #ffffff;
            animation: heartbeat 1.5s infinite;">    
            Developed by Harsh Pandey
        </p>
    </div>
    <style>@keyframes heartbeat { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }</style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
