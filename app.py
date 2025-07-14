from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')

# Setup upload directory
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'model/final_model.h5')  # atau model.keras jika kamu simpan ulang
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

# Load label map
try:
    with open("model/label_map.json", "r") as f:
        label_map = json.load(f)
except Exception as e:
    raise RuntimeError(f"Gagal memuat label map: {e}")

# Reverse mapping: index -> label
label_map_rev = {v: k for k, v in label_map.items()}
sorted_label_rev = [label_map_rev[i] for i in range(len(label_map))]

# Fungsi validasi ekstensi
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'Tidak ada file yang diupload.', 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return 'File tidak valid.', 400

    # Simpan file dengan nama unik
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Proses gambar
    try:
        img = Image.open(filepath).resize((128, 128)).convert('RGB')
    except Exception:
        return 'Gagal membaca gambar. Pastikan file berupa gambar yang valid.', 400

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(prediction)
    result_label = sorted_label_rev[pred_class_idx]

    # Rekomendasi tindakan
    if result_label == "daur_ulang":
        advice = "♻️ Sampah ini sebaiknya didaur ulang. Contohnya bisa dijadikan kerajinan tangan."
    elif result_label in ["organik", "anorganik", "b3"]:
        advice = f"🗑️ Sampah ini tergolong {result_label.upper()}, sebaiknya dibuang ke tong yang sesuai."
    else:
        advice = "❓ Jenis sampah tidak dikenali."

    image_url = url_for('static', filename=f'uploads/{filename}')
    return render_template('index.html', prediction=result_label, image_url=image_url, advice=advice)

if __name__ == '__main__':
    app.run(debug=True)  # Ganti jadi False jika untuk deploy
