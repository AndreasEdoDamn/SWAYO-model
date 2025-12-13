import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2
import numpy as np

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="SWAYO – Smart Waste YOLO",
    layout="wide",
    page_icon="🗑️"
)

# ============================
# CSS
# ============================
st.markdown("""
<style>
body { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); }

.main-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    margin-top: 30px;
    text-align: center;
}

h1 {
    font-weight: 700 !important;
    color: #febd14 !important;
}

.video-wrapper {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

a {
    text-decoration: none;
    color: #febd14 !important;
    font-weight: 600;
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODEL
# ============================
model = YOLO("best.torchscript")

# ============================
# VIDEO TRANSFORMER
# ============================
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, imgsz=640)
        annotated = results[0].plot()
        return annotated

# ============================
# HEADER
# ============================
st.markdown('<a href="https://swayo.vercel.app/categories.html">← Go Back</a>', unsafe_allow_html=True)
st.markdown("<h1>🗑️ SWAYO: Smart Waste Classifier with YOLO</h1>", unsafe_allow_html=True)

# ============================
# MAIN CONTENT
# ============================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.subheader("🎥 Real-time Object Detection")
st.write("Model akan mendeteksi sampah dari webcam secara real-time.")

st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)

webrtc_streamer(
    key="yolo-webcam",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown('</div></div>', unsafe_allow_html=True)

# ============================
# FOOTER
# ============================
st.markdown('<p style="text-align:center; margin-top:40px;">© 2025 SWAYO – Smart Waste Classifier with YOLO</p>', unsafe_allow_html=True)
