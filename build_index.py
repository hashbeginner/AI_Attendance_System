import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import zipfile
import pickle
import faiss
import numpy as np

from pathlib import Path
from deepface import DeepFace

# =========================
# PATHS
# =========================

ROOT = Path.cwd()

ARCHIVE_ZIP = ROOT / "archive.zip"
EXTRACT_PATH = ROOT / "archive"

DATASET_PATH = EXTRACT_PATH / "Dataset" / "Faces"

INDEX_FILE = ROOT / "face_index.faiss"
LABELS_FILE = ROOT / "labels.pkl"

# =========================
# AUTO EXTRACT
# =========================

if ARCHIVE_ZIP.exists() and not EXTRACT_PATH.exists():

    print("📦 Extracting archive.zip ...")

    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zip_ref:
        zip_ref.extractall(ROOT)

    print("✅ Extraction complete")

# =========================
# CHECK DATASET
# =========================

print(f"📂 Dataset path: {DATASET_PATH}")

if not DATASET_PATH.exists():
    raise Exception(f"❌ Dataset not found: {DATASET_PATH}")

# =========================
# STORAGE
# =========================

embeddings = []
labels = []

# =========================
# BUILD EMBEDDINGS
# =========================

for person_folder in DATASET_PATH.iterdir():

    if not person_folder.is_dir():
        continue

    person_name = person_folder.name

    print(f"\n👤 Processing: {person_name}")

    image_files = list(person_folder.glob("*"))

    for img_path in image_files:

        try:

            result = DeepFace.represent(
                img_path=str(img_path),
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=True,
                align=True
            )

            if not result:
                continue

            embedding = result[0]["embedding"]

            embeddings.append(embedding)
            labels.append(person_name)

            print(f"✔ {img_path.name}")

        except Exception as e:

            print(f"❌ Skipped {img_path.name}")

# =========================
# FINAL CHECK
# =========================

if len(embeddings) == 0:
    raise Exception("❌ No embeddings generated")

print(f"\n🧠 Total embeddings: {len(embeddings)}")

# =========================
# NUMPY
# =========================

embeddings = np.array(embeddings).astype("float32")

# IMPORTANT
faiss.normalize_L2(embeddings)

# =========================
# BUILD INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# =========================
# SAVE
# =========================

faiss.write_index(index, str(INDEX_FILE))

with open(LABELS_FILE, "wb") as f:
    pickle.dump(labels, f)

print("\n🎯 INDEX BUILD COMPLETE")
print("✔ face_index.faiss saved")
print("✔ labels.pkl saved")