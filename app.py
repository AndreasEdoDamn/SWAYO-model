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
}

h1 {
    font-weight: 700 !important;
    color: #febd14 !important;
}

.count-box {
    background: #f9f9f9;
    padding: 20px;
    border-radius: 14px;
    border-left: 5px solid #febd14;
    font-size: 18px;
    margin-bottom: 12px;
}

.count-title {
    font-size: 20px;
    font-weight: 700;
    color: #444;
}

.upload-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    margin-top: 30px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
}

img {
    max-width: 100%;
    border-radius: 12px;
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
# LOAD YOLO MODEL
# ============================
model = YOLO("best.torchscript")

# ============================
# VIDEO TRANSFORMER CLASS
# ============================
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.latest_counts = {}  # real-time count

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, imgsz=640)

        counts = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        self.latest_counts = counts
        annotated = results[0].plot()
        return annotated


# ============================
# HEADER
# ============================
st.markdown('<a href="https://swayo.vercel.app/categories.html">← Go Back</a>', unsafe_allow_html=True)
st.markdown("<h1>🗑️ SWAYO: Smart Waste Classifier with YOLO</h1>", unsafe_allow_html=True)


# ============================
# MAIN CARD
# ============================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.subheader("🎥 Real-time Object Detection")
st.write("Model akan mendeteksi sampah dari webcam secara real-time dan menghitung jumlah objek setiap detik.")

col1, col2 = st.columns([1.3, 0.7])

# ============================
# LEFT: VIDEO STREAM
# ============================
with col1:
    webrtc_ctx = webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )



# ============================
# FILE UPLOAD DETECTION
# ============================
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.subheader("📁 Upload Gambar untuk Deteksi")

uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.write("📌 Hasil deteksi:")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(img)

    # Annotated image
    annotated = results[0].plot()

    # Convert BGR → RGB
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Show image
    st.image(annotated, caption="Hasil Deteksi dengan Bounding Box")

    # Show Detection Details
    st.write("### 🔍 Detail Deteksi")
    counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    for cls, total in counts.items():
        st.markdown(
            f"""
            <div class="count-box">
                <div class="count-title">{cls.capitalize()}</div>
                Jumlah dalam gambar: <b>{total}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)  # close upload-card

# ============================
# FOOTER
# ============================
st.markdown('<p class="footer">© 2025 SWAYO – Smart Waste Classifier with YOLO</p>', unsafe_allow_html=True)

