# F:\keltron\projects\student_attendance_system\create_files.ps1

Write-Host "🚀 GENERATING PROJECT FILES..." -ForegroundColor Cyan

# --- 1. UTILS: GPU CHECKER ---
$gpu_check_code = @'
import onnxruntime as ort
import sys
import torch
import os

def check_gpu():
    print(f"🔍 [SYSTEM] Checking Hardware Acceleration...")
    
    # 1. Check Torch (CUDA)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ [GPU] DETECTED: {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("❌ [GPU] CRITICAL ERROR: Torch cannot see your GPU.")
        # We don't exit here because ONNX might still work, but it's bad news.

    # 2. Check ONNX Runtime (The Real Worker)
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        print("❌ [ONNX] CRITICAL: CUDAExecutionProvider NOT found!")
        print(f"   Available: {providers}")
        print("   The system will run on CPU (SLOW).")
        return False
    else:
        print("✅ [ONNX] CUDA Provider Available. AI will run on GPU.")
        return True
'@
Set-Content -Path "utils\gpu_check.py" -Value $gpu_check_code -Encoding UTF8


# --- 2. FRONTEND: STYLES (CSS) ---
$css_code = @'
/* Modern Dark Theme */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
div.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #FF0000;
    transform: scale(1.02);
}
.reportview-container .main .block-container {
    padding-top: 2rem;
}
h1 {
    color: #FF4B4B !important;
    text-align: center;
    font-family: 'Helvetica Neue', sans-serif;
}
.status-box {
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid #333;
    background: #1E1E1E;
}
'@
Set-Content -Path "frontend\styles.css" -Value $css_code -Encoding UTF8


# --- 3. AI ENGINE (CORE LOGIC) ---
$engine_code = @'
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# PATHS
# InsightFace will auto-download models to this directory on first run
MODEL_ROOT = os.path.abspath("./ai_models")
DB_PATH = "./database/embeddings.pkl"
SIMILARITY_THRESHOLD = 0.5

class SentinelEngine:
    def __init__(self):
        print("⏳ [AI] Initializing InsightFace (GPU)...")
        # Initialize FaceAnalysis
        # providers=['CUDAExecutionProvider'] forces GPU usage
        # root=MODEL_ROOT tells it where to store/find models
        self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT, providers=['CUDAExecutionProvider'])
        
        # ctx_id=0 means use GPU 0. det_size is detection resolution (640x640 is standard)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_embeddings = []
        self.known_names = []
        self.load_embeddings()
        print("✅ [AI] System Ready.")

    def load_embeddings(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data['embeddings']
                    self.known_names = data['names']
                    print(f"📂 [DB] Loaded {len(self.known_names)} faces.")
            except Exception as e:
                print(f"⚠️ [DB] Corrupt database, starting fresh: {e}")
                self.known_embeddings = []
                self.known_names = []
        else:
            print("🆕 [DB] No database found. Starting fresh.")

    def save_embeddings(self):
        # Ensure database folder exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, 'wb') as f:
            pickle.dump({'embeddings': self.known_embeddings, 'names': self.known_names}, f)

    def process_frame(self, frame):
        """
        Detects faces and recognizes them.
        Returns: List of Results [{name, bbox, confident}]
        """
        # InsightFace expects BGR image (standard OpenCV)
        faces = self.app.get(frame)
        results = []
        
        for face in faces:
            name = "Unknown"
            status = "Unregistered"
            color = (0, 0, 255) # Red for Unknown
            
            if len(self.known_embeddings) > 0:
                # Compare this face embedding with all known embeddings
                sims = cosine_similarity([face.embedding], self.known_embeddings)[0]
                best_idx = np.argmax(sims)
                
                if sims[best_idx] > SIMILARITY_THRESHOLD:
                    name = self.known_names[best_idx]
                    status = "Recognized"
                    color = (0, 255, 0) # Green for Known
            
            results.append({
                "name": name,
                "status": status,
                "bbox": face.bbox.astype(int),
                "color": color
            })
            
        return results

    def register_new_face(self, frame, new_name):
        """
        Strict Registration Logic:
        - 1 Unknown Face + Any number of Known Faces -> OK
        - >1 Unknown Faces -> BLOCKED
        """
        faces = self.app.get(frame)
        
        if len(faces) == 0:
            return False, "❌ No face detected."

        unknown_faces = []
        
        for face in faces:
            is_known = False
            if len(self.known_embeddings) > 0:
                sims = cosine_similarity([face.embedding], self.known_embeddings)[0]
                if np.max(sims) > SIMILARITY_THRESHOLD:
                    is_known = True
            
            if not is_known:
                unknown_faces.append(face)

        # SCENARIO CHECKS
        if len(unknown_faces) == 0:
            return False, "⚠️ Only already registered faces detected."
        
        if len(unknown_faces) > 1:
            return False, "❌ MULTIPLE UNIDENTIFIED PEOPLE! Only one new person allowed."

        # Register the single unknown face
        target_face = unknown_faces[0]
        self.known_embeddings.append(target_face.embedding)
        self.known_names.append(new_name)
        self.save_embeddings()
        
        # Save reference image
        save_dir = f"./data/students/{new_name}"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/ref_img.jpg", frame)
        
        return True, f"✅ Successfully registered {new_name}!"
'@
Set-Content -Path "face_engine.py" -Value $engine_code -Encoding UTF8


# --- 4. MAIN APP (STREAMLIT UI) ---
$app_code = @'
import os
import sys
import site

# -----------------------------------------------------------------------------
# 🔧 AUTO-CONFIGURE GPU (The "No-Install" Hack)
# -----------------------------------------------------------------------------
def setup_gpu_paths():
    if sys.platform == 'win32':
        try:
            libs_dir = site.getsitepackages()[0]
            nvidia_dir = os.path.join(libs_dir, 'nvidia')
            if os.path.exists(nvidia_dir):
                for root, dirs, files in os.walk(nvidia_dir):
                    if 'bin' in dirs:
                        bin_path = os.path.join(root, 'bin')
                        os.add_dll_directory(bin_path)
                        os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
        except Exception as e:
            print(f"❌ GPU Setup Error: {e}")

setup_gpu_paths()
# -----------------------------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime
from face_engine import SentinelEngine
from utils.gpu_check import check_gpu

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentinel AI", layout="wide", page_icon="🔒")

# Load CSS
with open('frontend/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- INIT SESSION STATE ---
if 'engine' not in st.session_state:
    with st.spinner("🚀 Booting AI Engine on GPU... (Downloading models if first run)"):
        st.session_state.engine = SentinelEngine()

# --- DATABASE ---
def init_db():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, session TEXT)''')
    conn.commit()
    conn.close()

def mark_attendance(name):
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    c.execute("SELECT time, session FROM attendance WHERE name=? AND date=?", (name, date_str))
    records = c.fetchall()
    
    msg, status = "", ""
    
    if not records:
        c.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", (name, date_str, time_str, 'Morning'))
        msg = f"🌅 Morning Attendance Marked: {name}"
        status = "success"
    else:
        # Check for Afternoon logic (simplified for demo)
        sessions = [r[1] for r in records]
        if 'Morning' in sessions and 'Afternoon' not in sessions:
            last_time = datetime.strptime(f"{date_str} {records[-1][0]}", "%Y-%m-%d %H:%M:%S")
            hours_diff = (now - last_time).total_seconds() / 3600
            
            if hours_diff < 2:
                msg = f"⏳ Too early for Afternoon. Wait {2-hours_diff:.1f} hrs."
                status = "warning"
            else:
                c.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", (name, date_str, time_str, 'Afternoon'))
                msg = f"☀️ Afternoon Attendance Marked: {name}"
                status = "success"
        else:
            msg = f"✅ Attendance Complete for {name}"
            status = "info"
            
    conn.commit()
    conn.close()
    return msg, status

init_db()

# --- SIDEBAR ---
st.sidebar.title("Sentinel AI 🔒")
st.sidebar.success("GPU ACCELERATED")
menu = st.sidebar.radio("Mode", ["Live Attendance", "Register Student", "Database"])

# --- MODE 1: LIVE ATTENDANCE ---
if menu == "Live Attendance":
    st.header("🔴 Live Surveillance")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        run = st.checkbox("START CAMERA", value=True)
        frame_window = st.image([])
    with col2:
        st.subheader("Activity Log")
        log_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found!")
            break
            
        # Process
        results = st.session_state.engine.process_frame(frame)
        
        # Draw
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            color = res['color']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, res['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if res['status'] == "Recognized":
                msg, stat = mark_attendance(res['name'])
                if stat == "success":
                    log_placeholder.success(msg)
                elif stat == "warning":
                    log_placeholder.warning(msg)

        # Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()

# --- MODE 2: REGISTER ---
elif menu == "Register Student":
    st.header("👤 New Student Registration")
    st.info("Ensure ONLY the new student is visible.")
    
    name_in = st.text_input("Enter Full Name / ID")
    img_buffer = st.camera_input("Capture Face")
    
    if img_buffer and name_in:
        bytes_data = img_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        success, msg = st.session_state.engine.register_new_face(cv2_img, name_in)
        if success:
            st.balloons()
            st.success(msg)
        else:
            st.error(msg)

# --- MODE 3: DATABASE ---
elif menu == "Database":
    st.header("📊 Attendance Records")
    conn = sqlite3.connect('database/attendance.db')
    df = pd.read_sql("SELECT * FROM attendance", conn)
    st.dataframe(df, use_container_width=True)
    conn.close()
'@
Set-Content -Path "app.py" -Value $app_code -Encoding UTF8

Write-Host "✅ ALL FILES GENERATED SUCCESSFULLY." -ForegroundColor Green
Write-Host "👉 NOW Run: .\start_app.bat"