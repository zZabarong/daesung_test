# app.py - 메인 Flask 애플리케이션

from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import cv2
import numpy as np
import os
import datetime
import sqlite3
import base64
import json
from face_recognition_system import FaceRecognitionSystem

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'attendance.db'


face_system = FaceRecognitionSystem(database_path='students_database')



def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()


    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    )
    ''')

    conn.commit()
    conn.close()


def save_attendance(student_id, date, time):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM attendance WHERE student_id=? AND date=?", (student_id, date))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                       (student_id, date, time))
        conn.commit()
        result = True
    else:
        result = False

    conn.close()
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    today = datetime.date.today().strftime('%Y-%m-%d')
    cursor.execute('''
    SELECT s.student_id, s.name, a.time 
    FROM students s 
    LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ? 
    ORDER BY s.name
    ''', (today,))

    students = cursor.fetchall()

    total_students = len(students)
    present_students = sum(1 for student in students if student['time'] is not None)

    conn.close()

    return render_template('dashboard.html',
                           students=students,
                           today=today,
                           total_students=total_students,
                           present_students=present_students)


@app.route('/students')
def manage_students():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students ORDER BY name")
    students = cursor.fetchall()

    conn.close()

    return render_template('students.html', students=students)


@app.route('/students/new', methods=['GET', 'POST'])
def new_student():
    if request.method == 'POST':
        student_id = request.form['student_id']
        name = request.form['name']

        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO students (student_id, name) VALUES (?, ?)",
                           (student_id, name))
            conn.commit()
            conn.close()

            # 학생 ID에 해당하는 디렉토리 생성
            student_dir = os.path.join(face_system.database_path, student_id)
            if not os.path.exists(student_dir):
                os.makedirs(student_dir)

            return redirect(url_for('capture_face', student_id=student_id))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('new_student.html', error="이미 존재하는 학생 ID입니다.")

    return render_template('new_student.html')


@app.route('/students/<student_id>/capture', methods=['GET', 'POST'])
def capture_face(student_id):
    if request.method == 'POST':
        image_data = request.form['image_data']
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_system.face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            student_dir = os.path.join(face_system.database_path, student_id)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, (x, y, w, h) in enumerate(faces):
                face_img = gray[y:y + h, x:x + w]
                img_path = os.path.join(student_dir, f"{timestamp}_{i}.jpg")
                cv2.imwrite(img_path, face_img)
            face_system.train_recognizer()

            return jsonify({"success": True, "faces_detected": len(faces)})
        else:
            return jsonify({"success": False, "message": "얼굴이 감지되지 않았습니다."})

    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE student_id=?", (student_id,))
    student = cursor.fetchone()

    conn.close()

    if student:
        return render_template('capture_face.html', student=student)
    else:
        return redirect(url_for('manage_students'))

@app.route('/students/<student_id>/delete', methods=['POST'])
def delete_student(student_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    cursor.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))

    conn.commit()
    conn.close()

    student_dir = os.path.join(face_system.database_path, student_id)
    if os.path.exists(student_dir):
        for file in os.listdir(student_dir):
            os.remove(os.path.join(student_dir, file))
        os.rmdir(student_dir)

    face_system.train_recognizer()

    return redirect(url_for('manage_students'))

@app.route('/attendance')
def view_attendance():
    date = request.args.get('date', datetime.date.today().strftime('%Y-%m-%d'))

    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
    SELECT a.id, a.student_id, s.name, a.date, a.time 
    FROM attendance a 
    JOIN students s ON a.student_id = s.student_id 
    WHERE a.date = ? 
    ORDER BY a.time
    ''', (date,))

    records = cursor.fetchall()

    conn.close()

    return render_template('attendance.html', records=records, current_date=date)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_system.face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            try:
                label, confidence = face_system.face_recognizer.predict(face)

                if confidence < 70:  # 신뢰도 임계값
                    student_id = face_system.labels_info[label]

                    conn = sqlite3.connect(app.config['DATABASE'])
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
                    result = cursor.fetchone()
                    conn.close()

                    if result:
                        name = result[0]

                        now = datetime.datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")

                        if save_attendance(student_id, date, time):
                            color = (0, 255, 0)
                            text = f"{name} - 출석 완료"
                        else:
                            color = (0, 165, 255)
                            text = f"{name} - 이미 출석함"
                    else:
                        color = (0, 0, 255)
                        text = "인식 오류"
                else:
                    color = (0, 0, 255)
                    text = "미등록"
            except Exception as e:
                color = (0, 0, 255)
                text = "오류"
                print(f"얼굴 인식 오류: {e}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    face_system.train_recognizer()
    app.run(debug=True)
