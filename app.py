import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import faiss
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import datetime
from deepface import DeepFace

# =========================
# PATHS
# =========================

ROOT        = Path.cwd()
UPLOADS_DIR = ROOT / "uploads"

ATTENDANCE_FILE = ROOT / "attendance.csv"
INDEX_FILE      = ROOT / "face_index.faiss"
LABELS_FILE     = ROOT / "labels.pkl"
UNKNOWN_LOG     = ROOT / "unknown_log.csv"

UNKNOWN_DIR = ROOT / "unknown_faces"
UNKNOWN_DIR.mkdir(exist_ok=True)

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="AI Attendance System", layout="wide")

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

index  = load_index()
labels = load_labels()

# =========================
# FILE SETUP
# =========================

UPLOADS_DIR.mkdir(exist_ok=True)

if not ATTENDANCE_FILE.exists():
    pd.DataFrame(columns=["Name", "Time", "Source"]).to_csv(ATTENDANCE_FILE, index=False)

if not UNKNOWN_LOG.exists():
    pd.DataFrame(columns=["Filename", "Timestamp"]).to_csv(UNKNOWN_LOG, index=False)

# =========================
# HELPERS
# =========================

def load_attendance():
    return pd.read_csv(ATTENDANCE_FILE)


def mark_attendance(name, source):
    if name in ["Unknown", "Unknown_Human", "No face detected", "Blurry Image", "Model files missing"]:
        return
    df = load_attendance()
    df.loc[len(df)] = [name, datetime.now().strftime("%H:%M:%S"), source]
    df.to_csv(ATTENDANCE_FILE, index=False)


def is_blurry(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return True
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < 100


def resize_image(img_path, max_width=800):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if w <= max_width:
        return
    scale = max_width / w
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(img_path), resized)


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
            fa       = face["facial_area"]
            cropped  = img[fa["y"]:fa["y"]+fa["h"], fa["x"]:fa["x"]+fa["w"]]
            filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
            cv2.imwrite(str(UNKNOWN_DIR / filename), cropped)
            pd.DataFrame([[filename, datetime.now()]]).to_csv(
                UNKNOWN_LOG, mode="a", header=False, index=False
            )
    except Exception as e:
        print("Unknown save error:", e)

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
        vector = np.array(result[0]["embedding"], dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector
    except:
        return None

# =========================
# SINGLE FACE RECOGNITION  (webcam)
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
    best_idx   = int(indices[0][0])
    if similarity < 0.55:
        save_unknown_face(img_path)
        return "Unknown"
    if best_idx < 0 or best_idx >= len(labels):
        return "Unknown"
    return labels[best_idx], similarity

# =========================
# MULTI-FACE RECOGNITION  (upload)
# =========================

def extract_faces_with_fallback(img_path):
    """Try retinaface → mtcnn → opencv. Return (faces, backend) or ([], None)."""
    for backend in ["retinaface", "mtcnn", "opencv"]:
        try:
            faces = DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend=backend,
                enforce_detection=True,
                align=True
            )
            if faces:
                return faces, backend
        except Exception:
            continue
    return [], None


def recognize_multiple_faces(img_path):
    """
    Returns:
        results      : list of (name: str, facial_area: dict, similarity: float)
        annotated_img: BGR numpy array with boxes/labels drawn, or None
    """
    if index is None or len(labels) == 0:
        return [], None

    img = cv2.imread(str(img_path))
    if img is None:
        return [], None

    faces, backend_used = extract_faces_with_fallback(img_path)
    if not faces:
        return [], None

    VALID_THRESHOLD = 0.55
    results = []

    for face_idx, face in enumerate(faces):
        label = f"Face {face_idx + 1}"
        try:
            rep = DeepFace.represent(
                img_path=face["face"],
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
                align=True
            )
            if not rep:
                results.append(("Unknown", face.get("facial_area", {}), 0.0))
                continue

            vector = np.array(rep[0]["embedding"], dtype="float32").reshape(1, -1)
            faiss.normalize_L2(vector)

            scores, indices = index.search(vector, 1)
            similarity = float(scores[0][0])
            best_idx   = int(indices[0][0])

            if similarity < VALID_THRESHOLD or best_idx < 0 or best_idx >= len(labels):
                results.append(("Unknown", face.get("facial_area", {}), similarity))
            else:
                results.append((labels[best_idx], face.get("facial_area", {}), similarity))

        except Exception as e:
            results.append(("Unknown", face.get("facial_area", {}), 0.0))

    # Draw bounding boxes on a copy
    annotated = img.copy()
    for name, fa, sim in results:
        if not fa:
            continue
        x, y, w, h = fa.get("x",0), fa.get("y",0), fa.get("w",0), fa.get("h",0)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        text = f"{name} ({sim:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(annotated, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return results, annotated

# =========================
# UI HEADER
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
    "📤 Upload", "📷 Webcam", "🎥 Video", "📋 Records", "📊 Dashboard"
])

# =========================
# TAB 1 — UPLOAD
# =========================

with tab1:

    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            path = UPLOADS_DIR / uploaded_file.name

            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            resize_image(path)

            st.markdown(f"**📁 {uploaded_file.name}**")

            # Reserve all slots BEFORE the spinner runs so layout
            # does not shift when results arrive (prevents shaking)
            img_slot     = st.empty()
            status_slot  = st.empty()
            results_slot = st.empty()

            # Show resized raw image immediately into reserved slot
            img_slot.image(str(path), width=700)

            with st.spinner("Analyzing faces..."):
                results, annotated_img = recognize_multiple_faces(path)

            if not results:
                status_slot.warning("⚠ No faces detected or model files missing.")
                st.markdown("---")
                continue

            # Replace raw image with annotated version in the same slot
            # width=700 matches resize_image max_width so no layout jump
            if annotated_img is not None:
                img_slot.image(
                    cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                    caption=f"Detected faces — {len(results)} found",
                    width=700
                )

            # Write all result text into the pre-reserved slot
            with results_slot.container():
                st.markdown("**Detected People:**")
                unknown_found = False
                unique_known  = set()

                for name, fa, sim in results:
                    if name == "Unknown":
                        unknown_found = True
                        st.error("❌ Unknown Human")
                    else:
                        unique_known.add(name)
                        st.success(f"✅ {name}  —  similarity: `{sim:.3f}`")

            status_slot.empty()  # clear the reserved status slot

            # Attendance + unknown logging
            if unknown_found:
                save_unknown_face(path)
                mark_attendance("Unknown_Human", "Upload")
                st.info("Unknown face stored.")

            for name in unique_known:
                mark_attendance(name, "Upload")

            st.markdown("---")

# =========================
# TAB 2 — WEBCAM
# =========================

with tab2:

    cam = st.camera_input("Take Photo")

    if cam:

        path = UPLOADS_DIR / "webcam.jpg"
        with open(path, "wb") as f:
            f.write(cam.getbuffer())

        resize_image(path)

        result = recognize_face(path)

        if isinstance(result, tuple):
            name, score = result
            st.success(f"✅ {name}")
            st.info(f"Similarity: {score:.3f}")
            mark_attendance(name, "Webcam")
        else:
            name = result
            st.warning(name)
            if name == "Unknown":
                mark_attendance("Unknown_Human", "Webcam")

# =========================
# TAB 3 — VIDEO
# =========================

with tab3:

    st.subheader("🎥 Video Attendance System")

    video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video:

        video_path = UPLOADS_DIR / video.name
        with open(video_path, "wb") as f:
            f.write(video.getbuffer())

        st.video(str(video_path))
        st.info("⏳ Processing video...")

        cap = cv2.VideoCapture(str(video_path))

        recognized            = set()
        frame_count           = 0
        frames_scanned        = 0
        total_face_detections = 0
        unknown_face_count    = 0
        person_first_seen     = {}
        face_counts_per_frame = []

        frame_skip   = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30

        progress_bar = st.progress(0)
        status_text  = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.text(
                    f"Frame {frame_count}/{total_frames}  |  "
                    f"Scanned: {frames_scanned}  |  Known: {len(recognized)}"
                )

            if frame_count % frame_skip != 0:
                continue

            frames_scanned += 1
            frame_resized   = cv2.resize(frame, (320, 240))

            try:
                faces = DeepFace.extract_faces(
                    img_path=frame_resized,
                    detector_backend="opencv",
                    enforce_detection=True
                )
                faces_in_frame         = len(faces)
                total_face_detections += faces_in_frame
                face_counts_per_frame.append((frame_count, faces_in_frame))

                for face in faces:
                    try:
                        rep = DeepFace.represent(
                            img_path=face["face"],
                            model_name="ArcFace",
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        if not rep:
                            unknown_face_count += 1
                            continue

                        vector = np.array(rep[0]["embedding"], dtype="float32").reshape(1, -1)
                        faiss.normalize_L2(vector)

                        scores, indices = index.search(vector, 1)
                        similarity = float(scores[0][0])
                        best_idx   = int(indices[0][0])

                        if similarity < 0.55 or best_idx < 0 or best_idx >= len(labels):
                            unknown_face_count += 1
                            continue

                        name = labels[best_idx]

                        if name not in person_first_seen:
                            person_first_seen[name] = frame_count

                        if name not in recognized:
                            recognized.add(name)
                            mark_attendance(name, "Video")

                    except Exception:
                        unknown_face_count += 1

            except Exception:
                face_counts_per_frame.append((frame_count, 0))

        cap.release()
        progress_bar.empty()
        status_text.empty()

        st.success("✅ Video Processing Complete")

        # Analytics
        st.subheader("📊 Video Analytics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Frames",       frame_count)
        m2.metric("Frames Scanned",     frames_scanned)
        m3.metric("Face Detections",    total_face_detections)
        m4.metric("Known People",       len(recognized))
        m5.metric("Unknown Detections", unknown_face_count)

        st.markdown("---")

        if face_counts_per_frame:
            st.subheader("📈 Face Activity Over Video")
            st.line_chart(
                pd.DataFrame(face_counts_per_frame, columns=["Frame", "Faces Detected"]).set_index("Frame")
            )

        st.markdown("---")

        st.subheader("👥 Known vs Unknown Breakdown")
        st.bar_chart(
            pd.DataFrame({
                "Category": ["Known", "Unknown"],
                "Count":    [len(recognized), unknown_face_count]
            }).set_index("Category")
        )

        st.markdown("---")

        if person_first_seen:
            st.subheader("🕐 First Appearance in Video")
            rows = []
            for person, frame_no in sorted(person_first_seen.items(), key=lambda x: x[1]):
                sec = round(frame_no / fps, 1)
                rows.append({"Name": person, "Frame": frame_no,
                              "Timestamp": f"{int(sec//60):02d}:{int(sec%60):02d}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")

        if recognized:
            st.subheader("✅ Recognized People")
            for person in sorted(recognized):
                st.success(f"✅ {person}")
        else:
            st.warning("⚠️ No known faces detected")

# =========================
# TAB 4 — RECORDS
# =========================

with tab4:
    st.dataframe(load_attendance())

# =========================
# TAB 5 — DASHBOARD
# =========================

with tab5:

    df = load_attendance()

    st.subheader("📊 Admin Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",  len(df))
    col2.metric("Unique People",  df["Name"].nunique() if len(df) > 0 else 0)
    col3.metric("Unknown Humans", len(df[df["Name"] == "Unknown_Human"]) if len(df) > 0 else 0)

    st.markdown("---")

    if len(df) > 0:
        st.subheader("👥 Top Attendees")
        st.bar_chart(df["Name"].value_counts().head(10))

    st.markdown("---")

    if len(df) > 0:
        st.subheader("📂 Attendance Source Breakdown")
        st.bar_chart(df["Source"].value_counts())

    st.markdown("---")

    # Unknown Face Log
    st.subheader("📄 Unknown Face Log")
    try:
        ulog = pd.read_csv(UNKNOWN_LOG, header=0)
        if not ulog.empty:
            st.dataframe(ulog)
        else:
            st.info("No unknown faces logged yet.")
    except Exception:
        st.info("No unknown faces logged yet.")

    st.markdown("---")

    # Unknown Face Gallery
    st.subheader("🕵️ Unknown Face Gallery")
    unknown_files = sorted(
        UNKNOWN_DIR.glob("*.jpg"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:10]

    st.metric("Stored Unknown Faces", len(list(UNKNOWN_DIR.glob("*.jpg"))))

    with st.expander("Show Unknown Face Gallery"):
        if unknown_files:
            cols = st.columns(4)
            for i, file in enumerate(unknown_files):
                with cols[i % 4]:
                    st.image(str(file), use_container_width=True)
        else:
            st.info("No unknown faces stored yet.")

    st.markdown("---")

    # Export
    st.subheader("⬇️ Export Records")
    st.download_button(
        label="Download Attendance CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="attendance.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Admin controls
    st.subheader("🧹 Admin Controls")
    if st.button("Clear Attendance Records"):
        pd.DataFrame(columns=["Name", "Time", "Source"]).to_csv(ATTENDANCE_FILE, index=False)
        st.success("Attendance records cleared.")
        st.rerun()
