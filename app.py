import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(
    page_title="Helmet Detection YOLOv8",
    layout="wide"
)

st.title("Helmet Detection YOLOv8")
st.write("Deteksi penggunaan helm dari gambar dan kamera secara realtime")

# Load model YOLOv8
model = YOLO("runs/detect/train/weights/best.pt")

# === Tab Upload Gambar ===
tab1, tab2 = st.tabs(["Upload Gambar", "Kamera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Inference langsung
        results = model(image, conf=0.25)
        result_img = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar Asli")
            st.image(image, width=500)

        with col2:
            st.subheader("Hasil Deteksi")
            st.image(result_img, width=500)

# === Tab Kamera ===
with tab2:
    st.write("Jalankan kamera di browser untuk deteksi realtime")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Resize supaya model lebih cepat
            img_resized = cv2.resize(img, (640, 640))

            # YOLOv8 inference
            results = model(
                img_resized,
                conf=0.25,
                iou=0.5,
                imgsz=640
            )

            annotated_frame = results[0].plot()
            return annotated_frame

    webrtc_streamer(
        key="helmet-detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )
