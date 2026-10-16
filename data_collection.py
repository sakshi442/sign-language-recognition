import os
import cv2
import numpy as np
from extract_landmarks import RealTimeKeypointExtractor

DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except OSError:
            pass

cap = cv2.VideoCapture(0)
extractor = RealTimeKeypointExtractor()

print("[INFO] Starting Data Collection... Press 'q' on the camera window anytime to stop.")

break_all = False

for action in actions:
    if break_all:
        break
    for sequence in range(no_sequences):
        if break_all:
            break
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            if not ret:
                break_all = True
                break

            frame, keypoints = extractor.process_frame(frame)

            if frame_num == 0:
                cv2.putText(frame, 'STARTING COLLECTION', (120, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting data for "{action}" Video #{sequence}', (15, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Sign Language Data Collection', frame)
                
                # Check for 'q' key during 2-second delay
                if cv2.waitKey(2000) & 0xFF == ord('q'):
                    break_all = True
                    break
            else: 
                cv2.putText(frame, f'Collecting data for "{action}" Video #{sequence}', (15, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Sign Language Data Collection', frame)

            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # Check for 'q' key during stream
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break_all = True
                break

# Proper Cleanup to release camera and close all windows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("[SUCCESS] Camera closed successfully!")