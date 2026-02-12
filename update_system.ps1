# F:\keltron\projects\student_attendance_system\update_system.ps1

Write-Host "🔄 UPGRADING SENTINEL AI..." -ForegroundColor Cyan

# --- 1. UPDATE ENGINE (Auto-Training Added) ---
$engine_code = @'
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Mute the spammy warnings
warnings.filterwarnings("ignore")

# PATHS
MODEL_ROOT = os.path.abspath("./ai_models")
DATA_ROOT = os.path.abspath("./data/students")
DB_PATH = "./database/embeddings.pkl"
SIMILARITY_THRESHOLD = 0.5

class SentinelEngine:
    def __init__(self):
        print("⏳ [AI] Initializing InsightFace (GPU)...")
        self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT, providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_embeddings = []
        self.known_names = []
        
        # 1. Load existing DB
        self.load_embeddings()
        
        # 2. SYNC with Data Folder (The New Feature)
        self.sync_data_folder()
        print("✅ [AI] System Ready.")

    def load_embeddings(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data['embeddings']
                    self.known_names = data['names']
            except:
                self.known_embeddings = []
                self.known_names = []

    def save_embeddings(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, 'wb') as f:
            pickle.dump({'embeddings': self.known_embeddings, 'names': self.known_names}, f)

    def sync_data_folder(self):
        """
        Scans /data/students folder.
        If a folder name exists but isn't in the DB, it learns the faces inside.
        """
        if not os.path.exists(DATA_ROOT):
            os.makedirs(DATA_ROOT, exist_ok=True)
            return

        print(f"🔄 [SYNC] Scanning {DATA_ROOT} for new faces...")
        new_faces_added = 0
        
        # Loop through every student folder
        for student_name in os.listdir(DATA_ROOT):
            student_path = os.path.join(DATA_ROOT, student_name)
            
            if not os.path.isdir(student_path):
                continue
                
            # Check if we already know this person
            if student_name in self.known_names:
                continue
                
            print(f"   👉 Learning new student: {student_name}")
            
            # Process images in that folder
            valid_embedding = None
            for img_file in os.listdir(student_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(student_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None: continue
                    
                    faces = self.app.get(img)
                    if len(faces) == 1:
                        valid_embedding = faces[0].embedding
                        break # Found a good face, stop scanning this folder
            
            if valid_embedding is not None:
                self.known_embeddings.append(valid_embedding)
                self.known_names.append(student_name)
                new_faces_added += 1
            else:
                print(f"   ⚠️ Could not find a clear face for {student_name}")

        if new_faces_added > 0:
            self.save_embeddings()
            print(f"✅ [SYNC] Added {new_faces_added} new students from files.")
        else:
            print("✅ [SYNC] Database is up to date.")

    def process_frame(self, frame):
        faces = self.app.get(frame)
        results = []
        for face in faces:
            name = "Unknown"
            status = "Unregistered"
            color = (0, 0, 255)
            
            if len(self.known_embeddings) > 0:
                sims = cosine_similarity([face.embedding], self.known_embeddings)[0]
                best_idx = np.argmax(sims)
                if sims[best_idx] > SIMILARITY_THRESHOLD:
                    name = self.known_names[best_idx]
                    status = "Recognized"
                    color = (0, 255, 0)
            
            results.append({
                "name": name,
                "status": status,
                "bbox": face.bbox.astype(int),
                "color": color
            })
        return results

    def register_new_face(self, frame, new_name):
        faces = self.app.get(frame)
        if len(faces) == 0: return False, "❌ No face detected."
        
        # Check strict "One Unknown" rule
        unknown_count = 0
        target_face = None
        
        for face in faces:
            is_known = False
            if len(self.known_embeddings) > 0:
                sims = cosine_similarity([face.embedding], self.known_embeddings)[0]
                if np.max(sims) > SIMILARITY_THRESHOLD: is_known = True
            
            if not is_known:
                unknown_count += 1
                target_face = face

        if unknown_count == 0: return False, "⚠️ Face already registered."
        if unknown_count > 1: return False, "❌ Multiple unknown people detected."

        self.known_embeddings.append(target_face.embedding)
        self.known_names.append(new_name)
        self.save_embeddings()
        
        save_dir = os.path.join(DATA_ROOT, new_name)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "ref_img.jpg"), frame)
        
        return True, f"✅ Registered {new_name}"
'@
Set-Content -Path "face_engine.py" -Value $engine_code -Encoding UTF8


# --- 2. UPDATE APP UI (Better Frontend) ---
$app_code = @'
import os
import sys
import site

# AUTO-CONFIGURE GPU
def setup_gpu_paths():
    if sys.platform == 'win32':
        try:
            libs_dir = site.getsitepackages()[0]
            nvidia_dir = os.path.join(libs_dir, 'nvidia')
            if os.path.exists(nvidia_dir):
                for root, dirs, files in os.walk(nvidia_dir):
                    if 'bin' in dirs:
                        os.add_dll_directory(os.path.join(root, 'bin'))
                        os.environ['PATH'] = os.path.join(root, 'bin') + os.pathsep + os.environ['PATH']
        except: pass
setup_gpu_paths()

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime
from face_engine import SentinelEngine

st.set_page_config(page_title="Sentinel AI", layout="wide", page_icon="👁️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
    h1, h2, h3 { color: #00ADB5 !important; }
    div[data-testid="stSidebar"] { background-color: #262730; }
</style>
""", unsafe_allow_html=True)

# --- INIT ENGINE ---
if 'engine' not in st.session_state:
    with st.spinner("🚀 Booting AI Engine & Syncing Data..."):
        st.session_state.engine = SentinelEngine()

# --- DB FUNCTIONS ---
def get_attendance_stats():
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(DISTINCT name) FROM attendance WHERE date=?", (today,))
    count = c.fetchone()[0]
    conn.close()
    return count

def mark_attendance(name):
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    
    c.execute("SELECT time, session FROM attendance WHERE name=? AND date=?", (name, date_str))
    records = c.fetchall()
    
    msg, status = "", ""
    if not records:
        c.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", (name, date_str, time_str, 'Morning'))
        msg, status = f"🌅 Morning Check-in: {name}", "success"
    else:
        sessions = [r[1] for r in records]
        if 'Morning' in sessions and 'Afternoon' not in sessions:
            last_time = datetime.strptime(f"{date_str} {records[-1][0]}", "%Y-%m-%d %H:%M:%S")
            if (now - last_time).total_seconds() / 3600 < 2:
                msg, status = f"⏳ Wait for Afternoon session.", "warning"
            else:
                c.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", (name, date_str, time_str, 'Afternoon'))
                msg, status = f"☀️ Afternoon Check-in: {name}", "success"
    
    conn.commit()
    conn.close()
    return msg, status

# --- UI LAYOUT ---
st.sidebar.title("👁️ Sentinel AI")
st.sidebar.info("running on NVIDIA CUDA")
page = st.sidebar.radio("Navigation", ["Live Monitor", "Registration Station", "Data Logs"])

if page == "Live Monitor":
    st.title("🔴 Real-Time Surveillance")
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("System Status", "Active", "GPU On")
    col2.metric("Date", datetime.now().strftime("%d %b %Y"))
    col3.metric("Present Today", get_attendance_stats())

    # Main Video Area
    col_vid, col_log = st.columns([2, 1])
    
    with col_vid:
        run = st.checkbox("Turn On Camera", value=True)
        frame_placeholder = st.empty()
    
    with col_log:
        st.subheader("Live Feed Logs")
        log_box = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret: break
        
        results = st.session_state.engine.process_frame(frame)
        
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), res['color'], 2)
            
            # Label with Background
            cv2.rectangle(frame, (x1, y1-30), (x2, y1), res['color'], -1)
            cv2.putText(frame, res['name'], (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            if res['status'] == "Recognized":
                msg, stat = mark_attendance(res['name'])
                if stat == "success": log_box.success(msg)
                elif stat == "warning": log_box.warning(msg)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()

elif page == "Registration Station":
    st.title("👤 New User Enrollment")
    st.info("System will auto-reject if multiple unknown people are seen.")
    
    c1, c2 = st.columns([1, 1])
    name_in = c1.text_input("Enter Full Name / Roll No")
    
    img_buffer = st.camera_input("Scan Face")
    
    if img_buffer and name_in:
        bytes_data = img_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        success, msg = st.session_state.engine.register_new_face(cv2_img, name_in)
        if success:
            st.balloons()
            st.success(msg)
        else:
            st.error(msg)

elif page == "Data Logs":
    st.title("📝 Attendance Records")
    conn = sqlite3.connect('database/attendance.db')
    df = pd.read_sql("SELECT * FROM attendance ORDER BY date DESC, time DESC", conn)
    st.dataframe(df, use_container_width=True)
'@
Set-Content -Path "app.py" -Value $app_code -Encoding UTF8


# --- 3. CREATE .GITIGNORE (GitHub Safe) ---
$gitignore = @'
# Ignore Portable Python
python_bin/
python_zip.zip
get-pip.py

# Ignore Large AI Models
ai_models/
*.onnx

# Ignore Database & Cache
database/
__pycache__/
*.pyc

# Ignore Virtual Env (if any)
venv/
env/
.env

# Ignore Logs
*.log
'@
Set-Content -Path ".gitignore" -Value $gitignore -Encoding UTF8

Write-Host "✅ UPGRADE COMPLETE!" -ForegroundColor Green
Write-Host "   1. Auto-Sync enabled (Check /data/students on startup)"
Write-Host "   2. New Dashboard UI installed"
Write-Host "   3. .gitignore created"
Write-Host "👉 Run '.\start_app.bat' to see the changes."