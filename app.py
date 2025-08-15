from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime
import face_recognition
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # Use env variable for production

# Paths (relative to app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_recognition_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "face_labels.txt")
STUDENTS_CSV = os.path.join(BASE_DIR, "models", "students.csv")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "models", "attendance.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Initialize model and labels
try:
    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    labels = []

# Load student data
def load_students_data():
    students_data = {}
    if os.path.exists(STUDENTS_CSV):
        df = pd.read_csv(STUDENTS_CSV)
        for _, row in df.iterrows():
            students_data[row["name"].lower()] = {
                "subject": row["subject"],
                "classroom": row["classroom"],
                "seat_no": str(row["seat_no"])
            }
    return students_data

# Dictionary to track recognition attempts
recognition_attempts = {}

# Function to track and manage recognition attempts
def manage_recognition_attempts(ip_address):
    """
    Track and manage recognition attempts for a specific IP address.
    Returns True if the user still has attempts remaining, False if attempts are exhausted.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Reset attempts if it's a new day
    if ip_address in recognition_attempts:
        if recognition_attempts[ip_address].get('date') != current_date:
            recognition_attempts[ip_address] = {'count': 0, 'date': current_date}
    else:
        recognition_attempts[ip_address] = {'count': 0, 'date': current_date}
    
    # Increment attempt count
    recognition_attempts[ip_address]['count'] += 1
    
    # Check if attempts are exhausted
    return recognition_attempts[ip_address]['count'] <= 3

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Student Login Page
@app.route('/student_login')
def student_login():
    return render_template('index.html')

# Admin Login Page
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid Credentials')
    return render_template('admin_login.html')

# Admin Dashboard
@app.route('/admin')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    students = read_csv(STUDENTS_CSV)
    attendance = read_csv(ATTENDANCE_CSV)
    return render_template('admin.html', students=students, attendance=attendance)

# Add Student
@app.route('/add_student', methods=['POST'])
def add_student():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    name = request.form['name']
    subject = request.form['subject']
    classroom = request.form['classroom']
    seat_no = request.form['seat_no']
    write_csv(STUDENTS_CSV, [name, subject, classroom, seat_no])
    return redirect(url_for('admin_dashboard'))

# Delete Student
@app.route('/delete_student', methods=['POST'])
def delete_student():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    name = request.form['name']
    delete_from_csv(STUDENTS_CSV, name)
    return redirect(url_for('admin_dashboard'))

# Upload Model function to handle both model.h5 and labels.txt
@app.route('/upload_model', methods=['POST'])
def upload_model():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    # Check if both files are in the request
    if 'model_file' not in request.files or 'labels_file' not in request.files:
        return redirect(url_for('admin_dashboard'))
    
    model_file = request.files['model_file']
    labels_file = request.files['labels_file']
    
    # Check if both files have valid filenames
    if model_file.filename == '' or labels_file.filename == '':
        return redirect(url_for('admin_dashboard'))
    
    # Check file extensions
    if model_file.filename.endswith('.h5') and labels_file.filename.endswith('.txt'):
        # Save model file
        model_file.save(MODEL_PATH)
        
        # Save labels file
        labels_file.save(LABELS_PATH)
        
        # Reload model and labels
        global model, labels
        try:
            model = joblib.load(MODEL_PATH)
            with open(LABELS_PATH, "r") as f:
                labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error reloading model or labels: {e}")
    
    return redirect(url_for('admin_dashboard'))

# CSV Utility Functions
def read_csv(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as file:
        return list(csv.reader(file))

def write_csv(filename, row):
    # Create the file with header if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write appropriate headers based on the file
            if filename == STUDENTS_CSV:
                writer.writerow(["name", "subject", "classroom", "seat_no"])
            elif filename == ATTENDANCE_CSV:
                writer.writerow(["name", "date", "timestamp", "status"])
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def update_csv(filename, old_value, new_data):
    rows = read_csv(filename)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            if row[0] == old_value:
                writer.writerow([new_data['name'], new_data['subject'], new_data['classroom'], new_data['seat_no']])
            else:
                writer.writerow(row)

def delete_from_csv(filename, value):
    rows = read_csv(filename)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            if row[0] != value:
                writer.writerow(row)

# Attendance Marking Function
def mark_attendance(student_name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_key = datetime.now().strftime("%Y-%m-%d")
    
    if os.path.exists(ATTENDANCE_CSV):
        df_attendance = pd.read_csv(ATTENDANCE_CSV)
        if ((df_attendance["name"] == student_name) & (df_attendance["date"] == date_key)).any():
            return False
    
    write_csv(ATTENDANCE_CSV, [student_name, date_key, timestamp, "Present"])
    return True

# Face Recognition Endpoint with face_recognition library
CONFIDENCE_THRESHOLD = 0.7

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Get user IP address for tracking attempts
    ip_address = request.remote_addr
    attempts_allowed = manage_recognition_attempts(ip_address)
    
    if not attempts_allowed:
        # Return a response that triggers a redirect on the client side
        return jsonify({
            "error": "Maximum recognition attempts exceeded. Access denied.",
            "redirect_to": url_for("home")
        }), 403
    
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Use face_recognition library to process the image
    try:
        # Load and process image
        image = face_recognition.load_image_file(filepath)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            os.remove(filepath)
            return jsonify({"error": "No face detected in the image"}), 400
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Use the first face found (assuming one person per image)
        face_encoding = face_encodings[0].reshape(1, -1)
        
        if model is None:
            os.remove(filepath)
            return jsonify({"error": "Model not loaded"}), 500
        
        # Predict with the model
        probabilities = model.predict_proba(face_encoding)[0]
        best_match_idx = np.argmax(probabilities)
        confidence_score = probabilities[best_match_idx] * 100
        
        predicted_label = labels[best_match_idx].lower() if best_match_idx < len(labels) else "unknown"
        
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": f"Error during face recognition: {str(e)}"}), 500
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
    
    students_data = load_students_data()
    
    if confidence_score/100 >= CONFIDENCE_THRESHOLD:
        # Reset attempts on successful recognition
        if ip_address in recognition_attempts:
            recognition_attempts[ip_address]['count'] = 0
        
        student_info = students_data.get(predicted_label, None)
        
        if student_info:
            attendance_marked = mark_attendance(predicted_label)
            
            response = {
                "name": predicted_label,
                "exam": student_info["subject"],
                "classroom": student_info["classroom"],
                "seat_no": student_info["seat_no"],
                "attendance": "Already Marked ✅" if not attendance_marked else "Marked ✅"
            }
        else:
            response = {"error": "Student not found", "message": "Face not recognized"}
    else:
        attempts_remaining = 3 - recognition_attempts[ip_address]['count']
        if attempts_remaining <= 0:
            response = {
                "error": "Face not recognized",
                "message": "Maximum attempts exceeded",
                "redirect_to": url_for("home")
            }
        else:
            response = {
                "error": "Face not recognized",
                "message": f"Attempt {recognition_attempts[ip_address]['count']} of 3. {attempts_remaining} attempts remaining."
            }
    
    return jsonify(response)

# Logout Admin
@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's PORT or fallback
    app.run(host='0.0.0.0', port=port, debug=False)  # Production settings
