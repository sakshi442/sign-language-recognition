import os
import cv2
import numpy as np
from extract_landmarks import RealTimeKeypointExtractor

RAW_DATA_PATH = 'Raw_Dataset'
OUTPUT_DATA_PATH = 'MP_Data'
SEQUENCE_LENGTH = 30

# Start with ONE category to test. Leave the list empty [] to process everything.
CATEGORIES = []

VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv')
extractor = RealTimeKeypointExtractor()


def get_videos(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXT)]


def process_video(vid_path, sample_dir):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, keypoints = extractor.process_frame(frame)
        frames.append(keypoints)
    cap.release()

    if len(frames) == 0:
        print(f"  [WARNING] Could not read: {vid_path}")
        return False

    if len(frames) < SEQUENCE_LENGTH:
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])
    elif len(frames) > SEQUENCE_LENGTH:
        idx = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
        frames = [frames[i] for i in idx]

    os.makedirs(sample_dir, exist_ok=True)
    for i in range(SEQUENCE_LENGTH):
        np.save(os.path.join(sample_dir, f"{i}.npy"), frames[i])
    return True


def process_class(class_label, class_dir):
    videos = get_videos(class_dir)
    print(f"[INFO] {class_label}: {len(videos)} videos")
    for idx, vid in enumerate(videos):
        sample_dir = os.path.join(OUTPUT_DATA_PATH, class_label, str(idx))
        # Skip clips that were already converted
        if os.path.isdir(sample_dir) and len(os.listdir(sample_dir)) >= SEQUENCE_LENGTH:
            continue
        process_video(os.path.join(class_dir, vid), sample_dir)


for category in sorted(os.listdir(RAW_DATA_PATH)):
    if CATEGORIES and category not in CATEGORIES:
        continue
    cat_dir = os.path.join(RAW_DATA_PATH, category)
    if not os.path.isdir(cat_dir):
        continue

    if get_videos(cat_dir):
        # Videos directly inside (like the adjective folders)
        process_class(category, cat_dir)
    else:
        # Sub-folders, one per word (like Animals/1. Dog)
        for word in sorted(os.listdir(cat_dir)):
            word_dir = os.path.join(cat_dir, word)
            if os.path.isdir(word_dir) and get_videos(word_dir):
                process_class(f"{category}_{word}", word_dir)

print("\n[SUCCESS] Done!")