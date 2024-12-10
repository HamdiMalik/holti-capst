from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import uvicorn
from tensorflow.keras.utils import get_file
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime

# Inisialisasi FastAPI
app = FastAPI()

# Inisialisasi model dan kelas prediksi
model = None
class_names = [
    'Bercak Daun',
    'Busuk Buah Antraknosa',
    'Kutu Daun',
    'Sehat',
    'Thrips',
    'Virus Kuning'
]

# Setup Firebase Admin
cred = credentials.Certificate("key-access/key-access-firestore.json")  # Ganti dengan path ke file kredensial Anda
firebase_admin.initialize_app(cred)

# Inisialisasi Firestore
db = firestore.client()

# Fungsi untuk memuat model dari URL
def load_model():
    global model
    model_url = "https://storage.googleapis.com/holti-bucket2024/model-ml/model.h5"
    
    try:
        # Unduh model dan simpan ke file lokal sementara
        model_path = get_file("model.h5", model_url)

        # Muat model dari file yang sudah diunduh
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Muat model saat aplikasi dimulai
load_model()

# Fungsi prediksi gambar
def predict_image(image: Image.Image):
    try:
        # Resize gambar ke (224, 224)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalisasi
        input_tensor = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch

        # Prediksi
        predictions = model.predict(input_tensor)
        predicted_class_index = np.argmax(predictions, axis=-1)[0]
        predicted_class = class_names[predicted_class_index]

        # Buat respons informatif
        confidence_percentage = predictions[0][predicted_class_index] * 100  # Ubah ke persentase
        return predicted_class, confidence_percentage
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
# Fungsi untuk menyimpan hasil prediksi ke Firestore
def save_to_firestore(prediction, confidence_percentage, timestamp):
    try:
        # Generate unique ID for document
        prediction_id = str(uuid.uuid4())

        # Data yang akan disimpan
        data = {
            "prediction": prediction,
            "confidence": float(confidence_percentage),  # Konversi ke float biasa
            "timestamp": timestamp,
        }

        # Simpan data ke koleksi Firestore
        db.collection("predictions").document(prediction_id).set(data)
        print(f"Prediction saved to Firestore with ID: {prediction_id}")
    except Exception as e:
        print(f"Error saving to Firestore: {str(e)}")

# Endpoint untuk prediksi
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # Baca file gambar
        image = Image.open(file.file)

        # Lakukan prediksi
        prediction, confidence_percentage = predict_image(image)

        # Ambil timestamp
        timestamp = datetime.utcnow()

        # Simpan hasil prediksi ke Firestore
        save_to_firestore(prediction, confidence_percentage, timestamp)

        # Respons yang lebih menarik
        response = {
            "status": "success",
            "message": "Prediksi berhasil dilakukan!",
            "prediction": prediction,
            "confidence": f"{confidence_percentage:.2f}%",  # Format ke 2 desimal
            "explanation": f"Berdasarkan analisis model, gambar ini diprediksi sebagai '{prediction}' dengan kepercayaan {confidence_percentage:.2f}%.",
        }

        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {e}")

# Jalankan server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)