import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import streamlit.components.v1 as components


# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="SWAYO – Smart Waste YOLO",
    layout="wide",
    page_icon="🗑️"
)


# ============================
# LOAD YOLO MODEL
# ============================
model = YOLO("best.torchscript")

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, imgsz=640)
        annotated = results[0].plot()
        return annotated


# ============================
# SIDEBAR NAVIGATION
# ============================
page = st.sidebar.radio(
    "Navigation",
    ["Webcam Detector", "Detection Page (HTML)"]
)


# ============================
# PAGE 1 — STREAMLIT WEBCAM YOLO
# ============================
if page == "Webcam Detector":
    st.markdown("<h1>🗑️ Smart Waste Classifier – Live YOLO Detection</h1>", unsafe_allow_html=True)
    st.write("Deteksi sampah real-time menggunakan webcam.")

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


# ============================
# PAGE 2 — RENDER FULL HTML
# ============================
if page == "Detection Page (HTML)":
    st.markdown("### 🌐 Halaman Detection (HTML Full)")
    st.write("Berikut tampilan HTML full seperti desain front-end kamu:")

    html_code = """
    <!-- MASUKKAN HTML YANG KAMU KIRIM TADI DI SINI -->
    <!-- Aku potong di sini biar pesan tidak kepanjangan -->
    """ + """

    """ + """

<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection - Smart Waste Classifier</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background: #f5f5f5;
            color: #333;
            min-height: 100vh;
        }

        /* Navbar */
        .navbar {
            background: linear-gradient(135deg, #2d7a6e 0%, #1a5950 100%);
            padding: 20px 60px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .nav-title {
            color: white;
            font-size: 20px;
            font-weight: 600;
            line-height: 1.4;
        }

        .nav-links {
            display: flex;
            gap: 40px;
            align-items: center;
        }

        .nav-links a {
            color: white;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            transition: opacity 0.3s;
        }

        .nav-links a:hover {
            opacity: 0.8;
        }

        .profile-btn {
            background: white;
            color: #2d7a6e;
            padding: 10px 25px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        /* Main Content */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 60px 60px;
        }

        /* Detection Card */
        .detection-card {
            background: white;
            border-radius: 20px;
            padding: 50px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 2px solid #e0e0e0;
        }

        h1 {
            font-size: 36px;
            font-weight: 700;
            color: #1a5950;
            margin-bottom: 40px;
        }

        /* Camera Preview Area */
        .camera-area {
            background: #f8f8f8;
            border: 3px dashed #d0d0d0;
            border-radius: 15px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
        }

        .camera-placeholder {
            text-align: center;
            color: #999;
        }

        .camera-placeholder svg {
            width: 100px;
            height: 100px;
            margin-bottom: 20px;
            opacity: 0.3;
        }

        #video {
            width: 100%;
            max-width: 100%;
            border-radius: 12px;
            display: none;
        }

        #canvas {
            width: 100%;
            max-width: 100%;
            border-radius: 12px;
            display: none;
        }

        .video-active {
            display: block !important;
        }

        /* Buttons */
        .button-group {
            display: flex;
            gap: 20px;
            align-items: center;
            justify-content: center;
            margin-bottom: 30px;
        }

        .btn {
            padding: 15px 40px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Poppins', sans-serif;
        }

        .btn-camera {
            background: #9e9e9e;
            color: white;
        }

        .btn-camera:hover {
            background: #757575;
            transform: translateY(-2px);
        }

        .btn-camera.active {
            background: #2d7a6e;
        }

        .btn-camera.active:hover {
            background: #1a5950;
        }

        .divider {
            font-size: 18px;
            font-weight: 600;
            color: #666;
        }

        /* File Upload */
        .file-upload-area {
            position: relative;
            display: inline-block;
        }

        .btn-upload {
            background: #9e9e9e;
            color: white;
        }

        .btn-upload:hover {
            background: #757575;
            transform: translateY(-2px);
        }

        #fileInput {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
            top: 0;
            left: 0;
        }

        .file-name {
            margin-top: 15px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }

        /* Results Section */
        .results-section {
            margin-top: 30px;
            padding: 25px;
            background: #f8f8f8;
            border-radius: 15px;
            display: none;
        }

        .results-section.show {
            display: block;
        }

        .results-section h2 {
            font-size: 22px;
            color: #2d7a6e;
            margin-bottom: 15px;
        }

        .detection-result {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
            border-left: 4px solid #2d7a6e;
        }

        .detection-result h3 {
            font-size: 18px;
            color: #1a5950;
            margin-bottom: 8px;
        }

        .detection-result p {
            color: #666;
            line-height: 1.6;
        }

        .confidence {
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            margin-top: 10px;
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            color: #2d7a6e;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2d7a6e;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .navbar {
                padding: 15px 30px;
                flex-direction: column;
                gap: 20px;
            }

            .container {
                padding: 40px 30px;
            }

            .detection-card {
                padding: 30px 20px;
            }

            h1 {
                font-size: 28px;
            }

            .button-group {
                flex-direction: column;
            }

            .divider {
                display: none;
            }
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar">
        <div class="nav-title">Smart Waste<br>Classifier With YOLO</div>
        <div class="nav-links">
            <a href="index.html">Home</a>
            <a href="detection.html">Detection</a>
            <a href="category.html">Category</a>
            <div class="profile-btn">nama profile</div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container">
        <div class="detection-card">
            <h1>Deteksi Jenis Sampah</h1>

            <!-- Camera Preview Area -->
            <div class="camera-area" id="cameraArea">
                <div class="camera-placeholder" id="placeholder">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 15.5c1.93 0 3.5-1.57 3.5-3.5S13.93 8.5 12 8.5 8.5 10.07 8.5 12s1.57 3.5 3.5 3.5zM9 2L7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/>
                    </svg>
                    <p>Area preview kamera atau gambar akan muncul di sini</p>
                </div>
                <video id="video" autoplay playsinline></video>
                <canvas id="canvas"></canvas>
            </div>

            <!-- Button Group -->
            <div class="button-group">
                <button class="btn btn-camera" id="cameraBtn">
                    📷 Buka Kamera
                </button>
                
                <span class="divider">Atau</span>
                
                <div class="file-upload-area">
                    <button class="btn btn-upload">
                        📁 Pilih file
                    </button>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
            </div>

            <div class="file-name" id="fileName"></div>

            <!-- Loading Animation -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Mendeteksi sampah...</p>
            </div>

            <!-- Results Section -->
            <div class="results-section" id="results">
                <h2>📊 Hasil Deteksi</h2>
                <div class="detection-result">
                    <h3>Plastik</h3>
                    <p>Sampah plastik terdeteksi dalam gambar. Plastik dapat didaur ulang melalui bank sampah atau TPS 3R terdekat.</p>
                    <span class="confidence">Confidence: 95%</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const cameraBtn = document.getElementById('cameraBtn');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const placeholder = document.getElementById('placeholder');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        let stream = null;
        let cameraActive = false;

        // Toggle Camera
        cameraBtn.addEventListener('click', async () => {
            if (!cameraActive) {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { facingMode: 'environment' } 
                    });
                    video.srcObject = stream;
                    video.classList.add('video-active');
                    placeholder.style.display = 'none';
                    canvas.style.display = 'none';
                    cameraBtn.textContent = '⏹️ Tutup Kamera';
                    cameraBtn.classList.add('active');
                    cameraActive = true;

                    // Simulate detection after 2 seconds
                    setTimeout(() => {
                        simulateDetection();
                    }, 2000);
                } catch (err) {
                    alert('Tidak dapat mengakses kamera. Pastikan Anda memberikan izin akses kamera.');
                    console.error('Error accessing camera:', err);
                }
            } else {
                // Stop camera
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                video.classList.remove('video-active');
                placeholder.style.display = 'block';
                cameraBtn.textContent = '📷 Buka Kamera';
                cameraBtn.classList.remove('active');
                cameraActive = false;
                results.classList.remove('show');
            }
        });

        // File Upload
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                fileName.textContent = `File terpilih: ${file.name}`;
                
                const reader = new FileReader();
                reader.onload = (event) => {
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        
                        // Show canvas
                        canvas.classList.add('video-active');
                        placeholder.style.display = 'none';
                        video.classList.remove('video-active');
                        
                        // Stop camera if active
                        if (stream) {
                            stream.getTracks().forEach(track => track.stop());
                            cameraActive = false;
                            cameraBtn.textContent = '📷 Buka Kamera';
                            cameraBtn.classList.remove('active');
                        }

                        // Simulate detection
                        simulateDetection();
                    };
                    img.src = event.target.result;
                };
                reader.readAsDataURL(file);
            }
        });

        // Simulate Detection (replace with actual YOLO detection)
        function simulateDetection() {
            loading.classList.add('show');
            results.classList.remove('show');

            setTimeout(() => {
                loading.classList.remove('show');
                results.classList.add('show');
                
                // Scroll to results
                results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 2000);
        }
    </script>
</body>
</html>

    """

    components.html(html_code, height=1800, scrolling=True)
