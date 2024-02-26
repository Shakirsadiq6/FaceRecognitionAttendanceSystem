# Face Recognition System with Flask
This project is a face recognition system implemented using Flask, OpenCV, and face_recognition library. The system allows users to upload images of employees, extracts their facial features, and stores the encoded data in a MongoDB database. Additionally, the system provides a recognition endpoint to mark attendance based on uploaded unknown faces.

### Prerequisites
Python 3.x<br>
Flask<br>
OpenCV<br>
face_recognition<br>
pymongo<br>

### Setup
Install dependencies using pip install -r requirements.txt.
Set up a MongoDB server and update the connection details in the code.

### Usage
Run app.py to start the web application. Visit http://localhost:5000 in your browser.
Upload images of employees to train the system.
Click the "Recognize" button to identify unknown faces.
Send a POST request to http://localhost:5000/mark_attendance with an image file to mark attendance based on recognized faces.

### Note
Ensure that you have the required cascades (haarcascade_frontalface_alt2.xml) for face detection.
Adjust file paths, directories, and database details as needed.
