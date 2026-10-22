import os
import cv2
import numpy as np
import mediapipe as mp
from static_utils import hands_to_features, landmarks_from_result

FOLDER = os.path.join('Raw_Dataset', 'Indian')
MAX_PER_CLASS = 300

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                 min_detection_confidence=0.3)

X, y = [], []
classes = sorted(d for d in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, d)))
print('[INFO] Classes:', classes)

for label in classes:
    d = os.path.join(FOLDER, label)
    files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_PER_CLASS]
    ok = 0
    for f in files:
        img = cv2.imread(os.path.join(d, f))
        if img is None:
            continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lm = landmarks_from_result(res)
        if lm:
            X.append(hands_to_features(lm))
            y.append(label)
            ok += 1
    print(f'{label}: {ok}/{len(files)} photos used')

np.savez('static_features.npz', X=np.array(X), y=np.array(y))
print('[SUCCESS] Saved static_features.npz with', len(X), 'samples')