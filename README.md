🎓 FaceCertify – Real-Time Exam Verification
==============================================================================

FaceCertify is an AI-powered facial recognition system designed to automate student verification and attendance during academic examinations. It provides a seamless, real-time identity check for students entering the exam hall, thereby enhancing academic integrity, eliminating impersonation, and saving administrative effort.

🚀 Built by Anson Mathew Jaison, Arjun Gireesh, Ebin Manoj, and Elwin Vincent  
👩‍🏫 Guided by Ms. Neha Beegam P.E.  
🏫 Department of Computer Science & Engineering, Viswajyothi College of Engineering and Technology (VJCET)

---

❓ Problem Statement
-------------------

Traditional manual verification processes are:

*   Time-consuming
*   Prone to impersonation and human error
*   Resource-intensive for both staff and students
    

✅ Our Solution
--------------

FaceCertify replaces manual ID checking with real-time facial verification using AI. The system:

*   Instantly validates a student’s face using an SVM-trained model
*   Auto-logs verified attendance into CSV files
*   Displays classroom and seat number after authentication
*   Offers an admin dashboard for managing the model and student records
    

✨ Key Features
--------------

*   Real-Time Face Recognition using dlib and face\_recognition
*   SVM classifier for student identification
*   Webcam or uploaded image capture  
*   Admin login to manage model and students 
*   Auto-generated attendance logs in CSV
*   Instant seat number display upon verification
*   Offline-ready, lightweight and fast
    

🛠️ Tech Stack
-------------------

| Layer       | Technology                   |
|-------------|-------------------------------|
| Backend     | Python 3, Flask               |
| AI Model    | face_recognition, dlib        |
| Classifier  | Scikit-Learn (SVM)            |
| Storage     | CSV for students & attendance |
| Frontend    | HTML, Tailwind CSS, JavaScript|
| Tools       | OpenCV, NumPy, Joblib, h5py   |
    

🗂️ Project Structure
-------------------

```plaintext
FaceCertify/
├── app.py                  # Main Flask application
├── requirements.txt
├── .gitignore
├── README.md
├── models/                 # AI models and data
│   ├── face_recognition_model.h5
│   ├── face_labels.txt
│   ├── attendance.csv
│   ├── students.csv
├── templates/              # Flask templates
│   ├── index.html
│   ├── admin_login.html
│   ├── admin.html
│   ├── home.html
├── uploads/                # Temporarily stores uploaded images
├── training/               # Training utilities
│   ├── train.py
│   ├── test.py
├── images/                 # Training data images
│   ├── Student1/
│   ├── Student2/
│   └── ...
├── index.html            # Project Portfolio

```

🚀 How to Run Locally
---------------------

To run FaceCertify on your local machine, follow the steps below:

1. **Clone the project from GitHub:**

```bash
git clone https://github.com/ansonjaison/FaceCertify.git
cd FaceCertify
```
    
2. **Create a virtual environment and activate it**

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**

 ```sh
 pip install -r requirements.txt
 ```
    
4. **Run the Flask app**

```sh
python app.py
```

5. **Access the Web Interface**
Open `http://127.0.0.1:5000/` in your browser.

🧠 How It Works
-------------------

1. **Admin logs in** and uploads a trained **SVM model**.
2. **Students approach the webcam** (or upload a photo).
3. **Face encoding is generated** and compared to trained encodings.
4. If **confidence ≥ 70%**, student is authenticated.
5. **Seat number + classroom is shown**. Attendance is logged.


📊 Data Formats
-------------------

### `students.csv`

| name           | subject | classroom | seat_no |
|----------------|---------|-----------|---------|
| Anson Jaison   | Math    | Room A    | 14      |


### `attendance.csv`

| name           | date       | timestamp | status |
|----------------|------------|-----------|--------|
| Anson Jaison   | 2025-07-24 | 08:45 AM  | ✔️     |


🧪 Training the AI Model
-------------------

Organize images like:

```
images/
├── Anson/
│   ├── img1.jpg
│   ├── img2.jpg
├── Arjun/
│   ├── img1.jpg
│   ├── img2.jpg
```

Then run:

```bash
cd training
python train.py
```

**Outputs:**

```
models/face_recognition_model.h5
models/face_labels.txt
```

You can test using:

```bash
python test.py
```


🛡️ Admin Credentials
-------------------

**Default:**

- **Username:** `admin`
- **Password:** `password`

(Located in `app.py` → `/admin_login` route. Can be changed.)
    

🚀 Future Scope
-------------------

- Integration with **MySQL** or **Supabase**
- Secure password hashing and **JWT login**
- **Admin-side model training** from GUI
- Real-time **seating visualization**
- **AI proctoring** & behavior monitoring
- **Mobile App Companion**


👨‍💻 Team Credits
-------------------

- **Anson Mathew Jaison**  
- **Arjun Gireesh**  
- **Ebin Manoj**  
- **Elwin Vincent**
- Under the guidance of **Ms. Neha Beegam P.E** *(CSE Dept., VJCET)*
    

📄 License
----------

This project is licensed under the MIT License.Feel free to use, modify, and share.

If you're an academic institution or startup interested in integrating FaceCertify into your systems, feel free to contact us via the GitHub Pages landing site or raise an issue in this repository.Feel free to fork, star, and contribute
