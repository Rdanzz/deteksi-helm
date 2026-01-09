import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Helmet Detection YOLOv8",
    layout="wide"
)

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

st.title("Helmet Detection YOLOv8")
st.write("Deteksi penggunaan helm dari gambar dan kamera secara realtime")

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

with tab2:
    run = st.checkbox("Jalankan Kamera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Tidak bisa membuka kamera")
            break

        frame_resized = cv2.resize(frame, (640, 640))

        results = model(
            frame_resized,
            conf=0.15,
            iou=0.5,
            imgsz=640
        )

        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(
            annotated_frame,
            channels="BGR",
            width=700
        )

    cap.release()
