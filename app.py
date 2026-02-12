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

st.set_page_config(page_title="Sentinel AI", layout="wide", page_icon="ðŸ‘ï¸")

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
    with st.spinner("ðŸš€ Booting AI Engine & Syncing Data..."):
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
        msg, status = f"ðŸŒ… Morning Check-in: {name}", "success"
    else:
        sessions = [r[1] for r in records]
        if 'Morning' in sessions and 'Afternoon' not in sessions:
            last_time = datetime.strptime(f"{date_str} {records[-1][0]}", "%Y-%m-%d %H:%M:%S")
            if (now - last_time).total_seconds() / 3600 < 2:
                msg, status = f"â³ Wait for Afternoon session.", "warning"
            else:
                c.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", (name, date_str, time_str, 'Afternoon'))
                msg, status = f"â˜€ï¸ Afternoon Check-in: {name}", "success"
    
    conn.commit()
    conn.close()
    return msg, status

# --- UI LAYOUT ---
st.sidebar.title("ðŸ‘ï¸ Sentinel AI")
st.sidebar.info("running on NVIDIA CUDA")
page = st.sidebar.radio("Navigation", ["Live Monitor", "Registration Station", "Data Logs"])

if page == "Live Monitor":
    st.title("ðŸ”´ Real-Time Surveillance")
    
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
    st.title("ðŸ‘¤ New User Enrollment")
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
    st.title("ðŸ“ Attendance Records")
    conn = sqlite3.connect('database/attendance.db')
    df = pd.read_sql("SELECT * FROM attendance ORDER BY date DESC, time DESC", conn)
    st.dataframe(df, use_container_width=True)
