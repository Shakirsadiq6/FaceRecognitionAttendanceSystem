from flask import Flask, request
import os, pickle, json
from pymongo import MongoClient
import face_recognition

__author__ = "Shakir Sadiq"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

client = MongoClient('localhost', 27017)
db = client['face_recognition']
col = db.employee_faces

def recognition(unknown_image):
    print("Loading encodings...")
    for record in db.pickle_data.find({}, {"_id": 0, "Pickle File": 1}):
        data = pickle.loads(record["Pickle File"])

    test_image = face_recognition.load_image_file(unknown_image)
    print("Recognizing faces...")
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)

        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = data["names"][first_match_index]
            result = {
                "Message": "Attendance marked for "+name
                }
        else:
            result = {
                "Message": "Unknown Face"
                }
        return result
    
app = Flask(__name__)

@app.route('/', methods= ['GET'])
def home():
    return "Hello! API is alive"

@app.route('/mark_attendance', methods= ['GET', 'POST'])
def handle_request():
    
    # image = 'Test_Images/1.jpg'
    image = request.files["image"]
    name = recognition(image)
    resp_data = {'name': name}

    return json.dumps(resp_data['name'])

if __name__ == "__main__":
    app.run(debug=True, port=1111)
