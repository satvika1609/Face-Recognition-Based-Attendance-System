from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
from datetime import datetime
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_face():
    os.system("python face_logic/face_register.py")
    return render_template("index.html", message="✅ Face registered successfully!")

@app.route('/extract')
def extract_features():
    os.system("python face_logic/extract_features.py")
    return render_template("index.html", message="✅ Features extracted successfully!")

@app.route('/attendance')
def start_attendance():
    subprocess.Popen(["python", "face_logic/attendance_runner.py"])
    return render_template("index.html", message="✅ Attendance completed successfully!")

@app.route('/records', methods=['GET', 'POST'])
def view_records():
    selected_date = ''
    attendance_data = []
    no_data = False

    if request.method == 'POST':
        selected_date = request.form.get('selected_date')
        selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
        formatted_date = selected_date_obj.strftime('%Y-%m-%d')

        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
        attendance_data = cursor.fetchall()
        conn.close()

        if not attendance_data:
            no_data = True

    return render_template('records.html', selected_date=selected_date, attendance_data=attendance_data, no_data=no_data)

if __name__ == '__main__':
    app.run(debug=True)
