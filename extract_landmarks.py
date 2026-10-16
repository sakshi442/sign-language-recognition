import cv2
import numpy as np
import mediapipe as mp

class RealTimeKeypointExtractor:
    def __init__(self):
        # MediaPipe Holistic Engine
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, results):
        """
        Extracts 50 keypoints (150 floats: x, y, z):
        - 8 Upper Body Landmarks
        - 21 Left Hand Landmarks
        - 21 Right Hand Landmarks
        """
        # 1. Upper Body Joints (8 points)
        pose_landmarks = []
        if results.pose_landmarks:
            selected_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # Shoulders, Elbows, Wrists, Hips
            for idx in selected_indices:
                lm = results.pose_landmarks.landmark[idx]
                pose_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            pose_landmarks = [0.0] * (8 * 3)

        # 2. Left Hand (21 points)
        lh_landmarks = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                lh_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            lh_landmarks = [0.0] * (21 * 3)

        # 3. Right Hand (21 points)
        rh_landmarks = []
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                rh_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            rh_landmarks = [0.0] * (21 * 3)

        return np.array(pose_landmarks + lh_landmarks + rh_landmarks, dtype=np.float32)

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        # Skeleton overlays
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        landmarks = self.extract_landmarks(results)
        return frame, landmarks

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    extractor = RealTimeKeypointExtractor()

    print("[INFO] Webcam is starting... press 'q' to exit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, keypoints = extractor.process_frame(frame)

        cv2.putText(frame, f"Extracted Points Shape: {keypoints.shape}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Keypoint Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()