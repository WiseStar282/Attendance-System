from flask import Flask, render_template, Response, request, redirect, url_for, jsonify  
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import mysql.connector
from datetime import date, datetime
from model.pipeline import *
import cv2
import random
import shutil
import os

app = Flask(__name__)
# MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Ganti dengan user MySQL Anda
    password="",  # Ganti dengan password MySQL Anda
    database="pegawai_db"
)
cursor = db.cursor()

app.config['UPLOAD_FOLDER'] = 'static'

# Direktori data    
train_dir = r"./Face Images/Training Images"
test_dir = r"./Face Images/Testing Images"

# Data augmentation dan data generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model/Model_Face_Recognition_mobilenet.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Inisial kamera
camera = cv2.VideoCapture(0)

def detect_faces_and_predict_labels(frame, model, class_indices):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    id_pegawai = None
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_preprocessed = preprocess_input(face_resized.astype(np.float32))
        face_preprocessed = np.expand_dims(face_preprocessed, axis=0)
        
        preds = model.predict(face_preprocessed)
        class_id = np.argmax(preds)
        confidence = preds[0][class_id]
        
        class_label = list(class_indices.keys())[class_id]
        id_pegawai = str(class_label)  # Mengonversi ke string
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_label}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame, id_pegawai

def generate_frames(detect_faces=True):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if detect_faces:
                frame, _ = detect_faces_and_predict_labels(frame, model, train_generator.class_indices)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(detect_faces=True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/raw_video_feed')
def raw_video_feed():
    return Response(generate_frames(detect_faces=False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        # Simpan frame asli tanpa bounding box
        img_name = f"static/captured_image.jpg"
        cv2.imwrite(img_name, frame)
        
        # Lakukan deteksi wajah dan prediksi
        frame, id_pegawai = detect_faces_and_predict_labels(frame, model, train_generator.class_indices)
        
        if id_pegawai is not None:
            today = date.today()
            tanggal_presensi = today.strftime("%Y-%m-%d")
            now = datetime.now()
            waktu_presensi = now.strftime("%H:%M:%S")
            
            berhasil_presensi = True
            
            cursor = db.cursor()
            cursor.execute("INSERT INTO pegawai (id_pegawai, tanggal_presensi, waktu_presensi, berhasil_presensi) VALUES (%s, %s, %s, %s)",
                           (id_pegawai, tanggal_presensi, waktu_presensi, berhasil_presensi))
            db.commit()
            cursor.close()
            
            # Redirect ke halaman hasil dengan data presensi
            return redirect(url_for('presensi', id_pegawai=id_pegawai, tanggal_presensi=tanggal_presensi, waktu_presensi=waktu_presensi))
    
    return "Failed to capture image"
    
@app.route('/presensi')
def presensi():
    tanggal_presensi = request.args.get('tanggal_presensi')
    waktu_presensi = request.args.get('waktu_presensi')
    nama_pegawai = request.args.get('id_pegawai')  
    
    return render_template('presensi.html', nama_pegawai=nama_pegawai, 
                           tanggal_presensi=tanggal_presensi, waktu_presensi=waktu_presensi, 
                           image_path="static/captured_image.jpg")
    
def index():
    render_template('/index.html')

@app.route('/pegawai_baru')
def pegawai_baru():
    return render_template('pegawai_baru.html')

@app.route('/ambil_images', methods=['GET'])
def ambil_images():
    success, frame = camera.read()
    if success:
        img_name = f"static/temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(img_name, frame)
        return jsonify({'image_path': img_name})
    return jsonify({'error': 'Gagal mengambil gambar'})

@app.route('/save_images', methods=['POST'])
def save_images():
    id_pegawai = request.form.get('id_pegawai')
    if not id_pegawai:
        return "ID Pegawai diperlukan", 400

    images = [f for f in os.listdir('static') if f.startswith('temp_image_')]
    if len(images) != 10:
        return "Harus ada tepat 10 gambar yang diambil", 400

    train_subdir = os.path.join(train_dir, id_pegawai)
    test_subdir = os.path.join(test_dir, id_pegawai)
    os.makedirs(train_subdir, exist_ok=True)
    os.makedirs(test_subdir, exist_ok=True)

    random.shuffle(images)
    for i, img in enumerate(images):
        src = os.path.join('static', img)
        if i < 8:
            dst = os.path.join(train_subdir, f"{id_pegawai}_{i}.jpg")
        else:
            dst = os.path.join(test_subdir, f"{id_pegawai}_{i}.jpg")
        shutil.move(src, dst)

    return redirect(url_for('konfirmasi_data'))

@app.route('/konfirmasi_data')
def konfirmasi_data():
    return render_template('konfirmasi_data.html')

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    # Panggil pipeline.py untuk melatih kembali model
    from model.pipeline import train_model
    train_model()
    return jsonify({'message': 'Retrain Model sukses!'})

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
