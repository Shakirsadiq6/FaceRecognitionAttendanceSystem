from flask import Flask, request, render_template, jsonify
import os, cv2, pickle, time
from pymongo import MongoClient
from bson.binary import Binary
import face_recognition

__author__ = "Shakir Sadiq"

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

client = MongoClient('localhost', 27017)
db = client['face_recognition']
col = db.employee_faces

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #"cnn"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)
if not os.path.exists("employee_images"):
    os.makedirs("employee_images")
if not os.path.exists("employee_images/original_images"):
    os.makedirs("employee_images/original_images")
if not os.path.exists("employee_images/cropped_images"):
    os.makedirs("employee_images/cropped_images")
if not os.path.exists("employee_images/unknown_images"):
    os.makedirs("employee_images/unknown_images")
if not os.path.exists("employee_images/B&W_images"):
    os.makedirs("employee_images/B&W_images")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    if request.form.get('encoding-button') == 'Upload':
        begin_upload =time.time()
        eid = request.form.get("eid") #Employee ID
        ename = request.form.get("ename") #Employee Name

        target = os.path.join(APP_ROOT, 'employee_images/original_images')

        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("file"):
            counter = 0
            imagename = "image{}.jpg"
            while os.path.exists('employee_images/original_images/'+imagename.format(counter)):
                counter += 1
            imagename = imagename.format(counter)
            destination = "/".join([target, imagename])
            file.save(destination)

            #haar cascade classifier
            faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            img_path=cv2.imread('employee_images/original_images/'+imagename)
            imgGray=cv2.cvtColor(img_path,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(imgGray,1.1,4)

            if len(faces) > 1:
                return "More than one faces detected, please! try another image."

            elif len(faces) == 0:
                return "No face detected, please! try another image."

            else:
                for (x,y,w,h) in faces:

                    crop_img = img_path[y:y+h, x:x+w]
                    cv2.imwrite('employee_images/cropped_images/crop'+imagename, crop_img)

                    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('employee_images/B&W_images/b&w'+imagename, gray_img)

                    resize_img = cv2.resize(gray_img, (384, 384))
                    dir_path = 'known_faces/'+ename
                    if not os.path.isdir(dir_path):
                        os.makedirs('known_faces/'+ename)
                    cv2.imwrite('known_faces/'+ename+'/resize'+imagename, resize_img)

                end_upload =time.time()
                print(f"Total runtime for uploading is {end_upload - begin_upload}")

                #Encoding
                begin_encoding = time.time()
                known_faces = []
                known_names = []

                for filename in os.listdir(f"{KNOWN_FACES_DIR}/"+ename):
                    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/"+ename+f"/{filename}")
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces.append(encoding)
                    known_names.append(ename)
                    
                #Storing Images into the DB
                if db.pickle_data.count_documents({}) == 0:
                    data = {"encodings": known_faces, "names": known_names}
                    encoded = pickle.dumps(data)
                    employee_data = {"Pickle File": encoded}
                    db.pickle_data.insert_one(employee_data)
                else:
                    for record in db.pickle_data.find({}, {"_id": 0, "Pickle File": 1}):
                        data = pickle.loads(record["Pickle File"])
                        data['encodings'].extend(known_faces)
                        data['names'].extend(known_names)
                        encoded = pickle.dumps(data)
                        db.pickle_data.delete_many({})
                        employee_data = {"Pickle File": encoded}
                        db.pickle_data.insert_one(employee_data)
                
                #Storing paths into the DB
                employee_original_image_path = 'employee_images/original_images/'+imagename
                employee_cropped_image_path = 'employee_images/cropped_images/crop'+imagename
                employee_BW_image_path = 'employee_images/B&W_images/b&w'+imagename
                employee_image_path = 'known_faces/'+ename+'/resize'+imagename
                employee_details = {
                    "Employee ID": eid,
                    "Employee Name": ename,
                    "Original Image Path": employee_original_image_path,
                    "Cropped Image Path": employee_cropped_image_path,
                    "B&W Image Path": employee_BW_image_path,
                    "Known Image Path": employee_image_path
                }
                db.employee_faces.insert_one(employee_details)
                
                end_encoding =time.time()
                print(f"Total runtime for encoding is {end_encoding - begin_encoding}")
                return "Data stored successfully!"

    elif request.form.get('recognise-button') == 'Recognise':
        begin_upload = time.time()
        target = os.path.join(APP_ROOT, 'employee_images/unknown_images')

        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("file"):
            counter = 0
            imagename = "image{}.jpg"
            while os.path.exists('employee_images/unknown_images/'+imagename.format(counter)):
                counter += 1
            imagename = imagename.format(counter)
            destination = "/".join([target, imagename])
            file.save(destination)

            #haar cascade classifier
            faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            img_path=cv2.imread('employee_images/unknown_images/'+imagename)
            imgGray=cv2.cvtColor(img_path,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(imgGray,1.1,4)

            if len(faces) > 1:
                return "More than one faces detected, please try another image."

            elif len(faces) == 0:
                return "No face detected, please try another image."

            else:
                for (x,y,w,h) in faces:

                    crop_img = img_path[y:y+h, x:x+w]
                    cv2.imwrite('employee_images/unknown_images/crop'+imagename, crop_img)

                    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('employee_images/unknown_images/b&w'+imagename, gray_img)

                    resize_img = cv2.resize(gray_img, (384, 384))
                    cv2.imwrite('unknown_faces/resize'+imagename, resize_img)

                end_upload =time.time()
                print(f"Total runtime for uploading is {end_upload - begin_upload}")

                #Recognition
                begin_recognition = time.time()
                
                #Loading encodings from DB
                print("Loading encodings...")
                for record in db.pickle_data.find({}, {"_id": 0, "Pickle File": 1}):
                    data = pickle.loads(record["Pickle File"])

                for filename in os.listdir(UNKNOWN_FACES_DIR):
                    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    print("Recognizing faces...")
                    boxes = face_recognition.face_locations(image, model=MODEL)
                    encodings = face_recognition.face_encodings(image, boxes)
                    names = []

                    for encoding in encodings:
                        matches = face_recognition.compare_faces(data["encodings"], encoding)

                        if True in matches:
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}
                            
                            for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1
                            print("Face recognized!")
                            result = {
                            "Message": "Face recognized as "+name
                            }
                            name = max(counts, key=counts.get)
                            names.append(name)
                            end_recognition =time.time()
                            print(f"Total runtime for recognising is {end_recognition - begin_recognition}")
                            os.remove(f"{UNKNOWN_FACES_DIR}/{filename}")
                            return jsonify(result), 200
                        else:
                            print("Face doesn't recognized!")
                            result = {
                            "Message": "Face not recognized"
                            }
                            os.remove(f"{UNKNOWN_FACES_DIR}/{filename}")
                            end_recognition =time.time()
                            print(f"Total runtime for recognising is {end_recognition - begin_recognition}")
                            return jsonify(result), 400

if __name__ == "__main__":
    app.run(debug=True)