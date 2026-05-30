# 🎯 AI Attendance System

## Overview

This project is an AI-powered attendance management system that uses facial recognition to automatically identify individuals and record their attendance. The system combines DeepFace, ArcFace, and FAISS to perform face detection, feature extraction, and fast similarity search.

The goal of this project is to eliminate manual attendance processes and demonstrate how facial recognition can be integrated into a practical real-world application.

---

## Features

### Image-Based Attendance

Users can upload an image containing a face. The system detects the face, compares it against the enrolled dataset, and records attendance automatically if a match is found.

### Webcam Attendance

Users can capture an image directly through their webcam and have their attendance recorded instantly.

### Video Processing

The system can process uploaded video files (MP4, AVI, MOV) and analyze frames to identify known individuals appearing in the video.

### Automatic Attendance Logging

Whenever a recognized individual is detected, the system automatically stores:

* Name
* Timestamp
* Detection Source (Upload, Webcam, or Video)

Attendance records are maintained in a CSV file for easy access and export.

### Unknown Human Detection

If a face does not belong to any enrolled individual, the system labels it as an unknown person instead of forcing an incorrect match.

### Unknown Face Storage

Images of unknown individuals are automatically stored for future review and possible enrollment into the dataset.

### Admin Dashboard

The dashboard provides:

* Total attendance records
* Unique individuals detected
* Number of unknown detections
* Attendance source analytics
* Attendance export functionality

### CSV Export

Attendance records can be downloaded as a CSV file directly from the application.

---

## Technologies Used

### DeepFace

Used for face detection and facial feature extraction.

### ArcFace

Used as the face recognition model to generate highly discriminative face embeddings.

### FAISS

Used for efficient similarity search and matching of facial embeddings.

### Streamlit

Used to build the user interface and deploy the application.

### OpenCV

Used for image and video processing.

### NumPy & Pandas

Used for numerical operations and attendance record management.

---

## System Workflow

1. User uploads an image, captures a webcam photo, or uploads a video.
2. Face detection is performed using DeepFace.
3. ArcFace generates a facial embedding.
4. The embedding is compared against stored embeddings using FAISS.
5. If similarity exceeds the defined threshold:

   * The individual is identified.
   * Attendance is recorded.
6. Otherwise:

   * The face is marked as unknown.
   * The image is stored for future review.

---

## Project Structure

AI_Attendance_System/

├── app.py

├── build_index.py

├── attendance.csv

├── face_index.faiss

├── labels.pkl

├── uploads/

├── unknown_faces/

├── archive.zip

├── requirements.txt

└── README.md

---

## Future Improvements

* Multi-face attendance tracking
* Real-time CCTV integration
* Liveness detection to prevent spoofing
* Cloud database integration
* User authentication and role-based access
* Attendance reports and analytics dashboard

---

## Conclusion

This project demonstrates the practical application of facial recognition for attendance management. By combining ArcFace embeddings, DeepFace processing, and FAISS similarity search, the system provides a fast and automated method of recording attendance while also handling unknown individuals safely.
