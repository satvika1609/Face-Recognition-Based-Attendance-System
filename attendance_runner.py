import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def create_attendance_table():
    """Ensure the attendance table exists in the database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            name TEXT,
            time TEXT,
            date DATE,
            UNIQUE(name, date)
        )
    """)
    conn.commit()
    conn.close()

class FaceRecognizer:
    def __init__(self):
        self.known_names = []
        self.known_features = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def load_known_faces(self):
        """Load known faces from the CSV features file."""
        features_path = "data/features_all.csv"
        if not os.path.exists(features_path):
            print("features_all.csv not found. Run feature extraction first.")
            return False

        df = pd.read_csv(features_path, header=None)
        self.known_names = df.iloc[:, 0].tolist()
        self.known_features = df.iloc[:, 1:129].astype(float).values

        print(f"[INFO] Loaded {len(self.known_names)} known face(s) from CSV.")
        return True

    def mark_attendance(self, name):
        """Log the attendance of a recognized person in the database."""
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, current_date))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)",
                (name, current_time, current_date)
            )
            print(f"[INFO] Attendance marked for {name} at {current_time}")

        conn.commit()
        conn.close()

    def recognize_faces(self):
        """Capture video stream, recognize faces and mark attendance."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not access the webcam.")
            return

        print("[INFO] Attendance mode started. Press 'q' to quit.")
        frame_count = 0
        skip_every = 2  
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            if frame_count % skip_every != 0:
                continue

            faces = detector(frame, 0)

            for face in faces:
                shape = predictor(frame, face)
                descriptor = face_reco_model.compute_face_descriptor(frame, shape)
                descriptor_np = np.array(descriptor)

                distances = np.linalg.norm(self.known_features - descriptor_np, axis=1)
                min_index = np.argmin(distances)
                min_distance = distances[min_index]

                name = "Unknown"
                if min_distance < 0.4:
                    name = self.known_names[min_index]
                    self.mark_attendance(name)

                # Draw bounding box and name
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def run():
    logging.basicConfig(level=logging.INFO)
    create_attendance_table()
    recognizer = FaceRecognizer()
    if recognizer.load_known_faces():
        recognizer.recognize_faces()

if __name__ == '__main__':
    run()
