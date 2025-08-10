import os
import cv2
import numpy as np

from pathlib import Path


def is_allowed_file(filename: str, allowed_extension) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in allowed_extension

# ================== Load Dataset ==================
def load_images_from_folder(path, image_size, color_mode="rgb"):
    # Check if the main path exists and is a directory
    if not os.path.exists(path):
        print("[ERROR] No such file or directory: '{}'".format(path))
        raise FileNotFoundError(f"[ERROR] Dataset does not exist")
    if not os.path.isdir(path):
        print("[ERROR] No such directory: '{}'".format(path))
        raise NotADirectoryError(f"[ERROR] Dataset path is not a directory")

    # Get subdirectories (assumed to be class folders)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if len(subdirs) < 2:
        raise ValueError(
            f"[ERROR] Insufficient class directories. Found {len(subdirs)} directories, but at least 2 are required.\n"
            f"Found directories: {subdirs}"
        )

    print(f"[INFO] Found {len(subdirs)} classes: {subdirs}")

    X, y = [], []
    label_names = sorted(subdirs)  # Ensure consistent ordering
    for label in label_names:
        folder_path = os.path.join(path, label)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if color_mode == "rgb":
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
                    img = cv2.resize(img, image_size)
                    X.append(img)
                    y.append(label)
            else:  # grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    X.append(img)
                    y.append(label)
    return np.array(X), np.array(y)