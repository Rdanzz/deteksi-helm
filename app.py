import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import av # Pastikan library 'av' sudah terinstall di requirements.txt
import numpy as np
# Tambahkan WebRtcMode di import
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

st.set_page_config(
    page_title="Helmet Detection YOLOv8",
    layout="wide"
)

st.title("Helmet Detection YOLOv8")
st.write("Deteksi penggunaan helm dari gambar dan kamera secara realtime")

# Load model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

tab1, tab2 = st.tabs(["Upload Gambar", "Kamera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
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
    st.write("Jalankan kamera di browser untuk deteksi realtime")

    class YOLOVideoProcessor:
        def recv(self, frame):
            # Konversi av.VideoFrame ke numpy array (BGR)
            img = frame.to_ndarray(format="bgr24")

            # Inference YOLOv8
            results = model(img, conf=0.25)
            
            # Gambar bounding box
            annotated_frame = results[0].plot()

            # Kembalikan sebagai av.VideoFrame
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Konfigurasi STUN Server
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="helmet-detection",
        video_processor_factory=YOLOVideoProcessor,
        mode=WebRtcMode.SENDRECV, # <--- PERBAIKAN DI SINI (Gunakan Enum, bukan string)
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )