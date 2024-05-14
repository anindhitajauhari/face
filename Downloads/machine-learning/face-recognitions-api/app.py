import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, Response
from flask.json import jsonify
from flask_cors import CORS
from retrain import Main
from facerec import FaceRec
import time
import os

UPLOAD_FOLDER = './videos'
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'jpeg', 'jpg', 'png'}

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

model_path = "facerec_model.clf"
Training = Main(model=model_path, source=0)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def folder_exists(username):
    dataset_path = os.path.join(os.getcwd(), "dataset")
    user_folder_path = os.path.join(dataset_path, username)
    return os.path.exists(user_folder_path)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Face Recognition API for Web Attendance',
        'version': '1.0',
        'status': "OK",
    })


"""
Route untuk mengupload data wajah
"""


@app.route("/generate-dataset", methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify('No file part'), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify('No selected file'), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Simpan file ke dalam folder videos
        timestamp = int(time.time())
        new_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(filepath)

        username = str(request.form['name']).lower()
        if not folder_exists(username):
            face_rec = FaceRec(username, filepath)
            face_rec.start()
            return jsonify('Dataset generated successfully'), 201
        else:
            return jsonify('Dataset already generated'), 409

    return jsonify('No file uploaded.'), 400


@app.route('/training', methods=['POST'])
def training():
    Training.training()
    return jsonify("Model trained successfully"), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify('No image sent'), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify('No selected file'), 400

    print(file)
    if file and allowed_file(file.filename):
        image_stream = file.read()  # Membaca file image
        nparr = np.fromstring(image_stream, np.uint8)  # Konversi ke numpy array
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode array menjadi frame
        Training.run(frame=frame)

    return jsonify([(name, Training.accuracy) for name, _ in Training.predictions])


@app.route('/check-facial-data/<name>', methods=['GET'])
def check_user_facial_data(name):
    username = str(name).lower()
    if not folder_exists(username):
        return jsonify('User folder not found'), 404
    else:
        return jsonify('User folder already exists'), 409
