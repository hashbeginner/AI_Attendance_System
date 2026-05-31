import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import faiss
import pickle
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import datetime
from deepface import DeepFace

# =========================
# PATHS
# =========================

ROOT = Path.cwd()

UPLOADS_DIR = ROOT / "uploads"

ATTENDANCE_FILE = ROOT / "attendance.csv"

INDEX_FILE = ROOT / "face_index.faiss"
LABELS_FILE = ROOT / "labels.pkl"


UNKNOWN_DIR = ROOT / "unknown_faces"
UNKNOWN_DIR.mkdir(exist_ok=True)

# =========================
# STREAMLIT
# =========================

st.set_page_config(
    page_title="AI Attendance System",
    layout="wide"
)

# =========================
# LOAD INDEX
# =========================

@st.cache_resource
def load_index():

    if not INDEX_FILE.exists():
        return None

    return faiss.read_index(str(INDEX_FILE))


@st.cache_data
def load_labels():

    if not LABELS_FILE.exists():
        return []

    with open(LABELS_FILE, "rb") as f:
        return pickle.load(f)

index = load_index()
labels = load_labels()

# =========================
# FILE SETUP
# =========================

UPLOADS_DIR.mkdir(exist_ok=True)

if not ATTENDANCE_FILE.exists():

    pd.DataFrame(
        columns=["Name", "Time", "Source"]
    ).to_csv(ATTENDANCE_FILE, index=False)

# =========================
# ATTENDANCE
# =========================

def load_attendance():
    return pd.read_csv(ATTENDANCE_FILE)


def mark_attendance(name, source):

    if name in [
        "Unknown",
        "No face detected",
        "Blurry Image",
        "Model files missing"
    ]:
        return

    df = load_attendance()

    df.loc[len(df)] = [
        name,
        datetime.now().strftime("%H:%M:%S"),
        source
    ]

    df.to_csv(ATTENDANCE_FILE, index=False)

# =========================
# BLUR DETECTION
# =========================

def is_blurry(img_path):

    img = cv2.imread(str(img_path))

    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return score < 100

# =========================
# EMBEDDING
# =========================

def get_embedding(img_path):

    try:

        result = DeepFace.represent(
            img_path=str(img_path),
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )

        if not result:
            return None

        embedding = result[0]["embedding"]

        vector = np.array(
            embedding,
            dtype="float32"
        ).reshape(1, -1)

        faiss.normalize_L2(vector)

        return vector

    except:
        return None
    
# =========================
# SAVE FUNCTION
# =========================
def save_unknown_face(img_path: Path):

    try:
        img = cv2.imread(str(img_path))

        if img is None:
            return

        faces = DeepFace.extract_faces(
            img_path=str(img_path),
            detector_backend="opencv",
            enforce_detection=True
        )

        for i, face in enumerate(faces):

            facial_area = face["facial_area"]

            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]

            cropped = img[y:y+h, x:x+w]

            filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"

            save_path = UNKNOWN_DIR / filename

            cv2.imwrite(str(save_path), cropped)

    except Exception as e:
        print("Unknown save error:", e)

# =========================
# RECOGNITION
# =========================
def recognize_face(img_path):

    if index is None or len(labels) == 0:
        return "Model files missing"

    if is_blurry(img_path):
        return "Blurry Image"

    query = get_embedding(img_path)

    if query is None:
        return "No face detected"

    scores, indices = index.search(query, 1)

    similarity = float(scores[0][0])

    best_idx = int(indices[0][0])

    VALID_THRESHOLD = 0.55

    if similarity < VALID_THRESHOLD:

        save_unknown_face(img_path)

        return "Unknown"

    if best_idx < 0 or best_idx >= len(labels):
        return "Unknown"

    return labels[best_idx], similarity
# =========================
# UI
# =========================

st.markdown(
    """
    <div style="text-align:center;">
        <h1>🎯 AI Attendance System</h1>
        <h4 style="color:gray;">⚡ FAISS + ArcFace + DeepFace</h4>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload",
    "📷 Webcam",
    "🎥 Video",
    "📋 Records",
    "📊 Dashboard"
])

# =========================
# IMAGE UPLOAD
# =========================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        path = UPLOADS_DIR / uploaded_file.name

        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(str(path))

        with st.spinner("Analyzing..."):

            result = recognize_face(path)

        if isinstance(result, tuple):

            name, score = result

            st.success(f"✅ {name}")
            st.info(f"Similarity: {score:.3f}")

            mark_attendance(name, "Upload")

        else:

            name = result

            if name == "Unknown":

                st.error("❌ Unknown Human")

                st.success(
                    "Unknown face stored successfully."
                )

                mark_attendance(
                    "Unknown_Human",
                    "Upload"
                )

            elif name == "No face detected":

                st.warning("⚠ No face detected")

            elif name == "Blurry Image":

                st.warning("⚠ Blurry Image")

            else:

                st.error(name)
# =========================
# WEBCAM
# =========================

with tab2:

    cam = st.camera_input("Take Photo")

    if cam:

        path = UPLOADS_DIR / "webcam.jpg"

        with open(path, "wb") as f:
            f.write(cam.getbuffer())

        st.image(str(path))

        result = recognize_face(path)

        if isinstance(result, tuple):

            name, score = result

            st.success(f"✅ {name}")
            st.info(f"Similarity: {score:.3f}")

        else:

            name = result

            st.warning(name)

        if name == "Unknown":
            mark_attendance("Unknown_Human", "Webcam")
        else:
            mark_attendance(name, "Webcam")

# =========================
# VIDEO
# =========================


with tab3:

    st.subheader("🎥 Video Attendance System")

    video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if video:

        video_path = UPLOADS_DIR / video.name

        with open(video_path, "wb") as f:
            f.write(video.getbuffer())

        st.video(str(video_path))

        st.info("⏳ Processing video...")

        cap = cv2.VideoCapture(str(video_path))

        recognized = set()

        frame_count = 0

        # ================= SPEED OPTIMIZATION =================
        frame_skip = 30   # process 1 frame every 30 frames

        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # ================= UPDATE PROGRESS =================
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)

            # ================= SKIP FRAMES =================
            if frame_count % frame_skip != 0:
                continue

            # ================= RESIZE FOR SPEED =================
            frame = cv2.resize(frame, (320, 240))

            try:

                # ================= FACE DETECTION =================
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="opencv",
                    enforce_detection=True
                )

                # ================= PROCESS EACH FACE =================
                for face in faces:

                    try:

                        face_img = face["face"]

                        # Convert face to embedding
                        result = DeepFace.represent(
                            img_path=face_img,
                            model_name="ArcFace",
                            detector_backend="skip",
                            enforce_detection=False
                        )

                        if not result:
                            continue

                        embedding = result[0]["embedding"]

                        vector = np.array(
                            embedding,
                            dtype="float32"
                        ).reshape(1, -1)

                        faiss.normalize_L2(vector)

                        # ================= SEARCH =================
                        scores, indices = index.search(vector, 1)

                        similarity = float(scores[0][0])

                        best_idx = int(indices[0][0])

                        VALID_THRESHOLD = 0.55

                        # ================= UNKNOWN =================
                        if similarity < VALID_THRESHOLD:
                            continue

                        if best_idx < 0 or best_idx >= len(labels):
                            continue

                        name = labels[best_idx]

                        # ================= DUPLICATE PREVENTION =================
                        if name in recognized:
                            continue

                        recognized.add(name)

                        mark_attendance(name, "Video")

                    except Exception:
                        continue

            except Exception:
                continue

        cap.release()

        progress_bar.empty()

        st.success("✅ Video Processing Complete")

        # ================= RESULTS =================
        if len(recognized) == 0:

            st.warning("⚠️ No known faces detected")

        else:

            st.subheader("👥 Recognized People")

            for person in recognized:
                st.success(f"✅ {person}")
# =========================
# RECORDS
# =========================

with tab4:

    st.dataframe(load_attendance())

# =========================
# DASHBOARD
# =========================

# ==================== DASHBOARD ====================

with tab5:

    df = load_attendance()

    st.subheader("📊 Admin Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        unique_people = df["Name"].nunique() if len(df) > 0 else 0
        st.metric("Unique People", unique_people)

    with col3:

        unknown_count = 0

        if len(df) > 0:
            unknown_count = len(
                df[df["Name"] == "Unknown_Human"]
            )

        st.metric("Unknown Humans", unknown_count)

    st.markdown("---")

    # ================= TOP ATTENDEES =================

    if len(df) > 0:

        st.subheader("👥 Top Attendees")

        top_people = df["Name"].value_counts().head(10)

        st.bar_chart(top_people)

    st.markdown("---")

    # ================= SOURCE ANALYTICS =================

    if len(df) > 0:

        st.subheader("📂 Attendance Source Breakdown")

        source_counts = df["Source"].value_counts()

        st.bar_chart(source_counts)

    st.markdown("---")
    # ============= UNKNOWN FACE GALLERY ============
    st.markdown("---")
    st.subheader("🕵️ Unknown Face Gallery")

    unknown_files = list(
        UNKNOWN_DIR.glob("*.jpg")
    )

    st.metric(
        "Stored Unknown Faces",
        len(unknown_files)
    )


    show_gallery = st.checkbox(
        "Show Unkown Face Gallery"
    )


    if show_gallery:
        cols = st.columns(4)

        for i, file in enumerate(unknown_files):

            with cols[i % 4]:
                st.image(
                    str(file),
                    use_container_width=True
                )
   
    # ================= CSV DOWNLOAD =================

    st.subheader("⬇️ Export Records")

    st.download_button(
        label="Download Attendance CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="attendance.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ================= CLEAR BUTTON =================

    st.subheader("🧹 Admin Controls")

    if st.button("Clear Attendance Records"):

        empty_df = pd.DataFrame(
            columns=["Name", "Time", "Source"]
        )

        empty_df.to_csv(ATTENDANCE_FILE, index=False)

        st.success("Attendance records cleared.")

        st.rerun()
