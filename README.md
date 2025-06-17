# Face Recognition Attendance System

This is a real-time face recognition-based attendance system built using Python, OpenCV, Dlib, and Flask. It provides a simple web interface for managing registration, feature extraction, attendance logging, and record viewing.

## Features

- Register new faces via webcam
- Extract facial features and store in CSV
- Recognize faces in real time and log attendance
- View attendance records by date using a web interface
- Attendance stored in SQLite database

## Tech Stack

- Python 3.10+
- OpenCV
- Dlib
- Flask (web framework)
- SQLite (local DB)
- HTML/CSS (Bootstrap 5)

## Setup Instructions

1. **Clone the repository**

   git clone https://github.com/satvika1609/face-recognition-attendance.git
   cd face-recognition-attendance


2. **Create a virtual environment (optional but recommended)**
 
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
 

3. **Install dependencies**

   pip install -r requirements.txt


4. **Ensure Dlib model files are present**

   Place the following files in `data/data_dlib/`:

   * `shape_predictor_68_face_landmarks.dat`
   * `dlib_face_recognition_resnet_model_v1.dat`

   These files are required for facial landmark detection and face encoding.

5. **Run the Flask application**

   python app.py
  

6. **Open your browser and visit**

   http://127.0.0.1:5000


## Acknowledgements

* Dlib library by Davis King
* OpenCV for real-time image processing
* Bootstrap for frontend styling
