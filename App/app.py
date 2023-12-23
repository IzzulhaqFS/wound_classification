import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
import keras.utils as image
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Konfigurasi upload file
UPLOAD_FOLDER = 'mysite/static/uploads'
ALLOWED_EXTENSION = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cek apakah ekstensi file diperbolehkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

# Load model yang digunakan
model = load_model('mysite/model-wound-recognition.h5', compile=False)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Fungsi untuk predict gambar
#def predict(image_path):
#    img = image.load_img(image_path, target_size=(32, 32))
#    img_array = image.img_to_array(img)
#    img_array = np.expand_dims(img_array, axis=0)
#    images = np.vstack([img_array])
#
#    prediction = model.predict(images)
#    prediction = np.argmax(prediction)
#    return prediction

@app.route('/')
def index():
    return "index.html"

@app.route('/api/upload', methods=['POST'])
def upload():
    data = {'wound' : "wound"}

    if 'image' not in request.files:
        data['wound'] = "File not found"
        return jsonify(data)

    file = request.files['image']

    if file.filename == '':
        data['wound'] = "Filename is null"
        return jsonify(data)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        #prediction = predict(filepath)
        #prediction = 5
        img = image.load_img(filepath, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        images = np.vstack([img_array])
        prediction = model.predict(images, batch_size=32)

        if prediction == 0:
            data['wound'] = 'Abrasions'
        elif prediction == 1:
            data['wound'] = 'Bruises'
        elif prediction == 2:
            data['wound'] = 'Burns'
        elif prediction == 3:
            data['wound'] = 'Cuts'
        elif prediction == 4:
            data['wound'] = 'Ingrown Nails'
        elif prediction == 5:
            data['wound'] = 'Laceration'
        else:
            data['wound'] = 'Stab Wounds'

        #data['wound'] = filepath
        #return jsonify(data)

    return jsonify(data)

if __name__ == '__main__':
    app.run()

