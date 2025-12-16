import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Pet Expression Classifier",
    page_icon="🐶",
    layout="centered"
)

# --- JUDUL & DESKRIPSI ---
st.title("🐶 Klasifikasi Ekspresi Hewan")
st.write("Upload gambar hewan peliharaan (Anjing/Kucing) untuk mendeteksi ekspresi: **Happy, Sad, atau Angry**.")

# --- LOAD MODEL (Dibuat Cache agar cepat) ---
@st.cache_resource
def load_trained_model():
    # Pastikan file .h5 ada satu folder dengan file app.py ini
    model = load_model('facial_expression_mobilenet.h5')
    return model

try:
    model = load_trained_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'facial_expression_mobilenet.h5' ada di direktori yang sama. Error: {e}")

# --- FUNGSI PREPROCESSING ---
# Harap diperhatikan: Preprocessing HARUS SAMA PERSIS dengan saat training
def preprocess_image(image):
    # 1. Konversi dari PIL Image (RGB) ke Numpy Array
    img_array = np.array(image)

    # 2. Konversi ke Grayscale
    # Catatan: PIL membaca sebagai RGB, cv2 butuh Gray
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # 3. Histogram Equalization (Penting: Sesuai training script Anda)
    img_eq = cv2.equalizeHist(img_gray)

    # 4. Resize ke 128x128
    img_resized = cv2.resize(img_eq, (128, 128))

    # 5. Normalisasi (0-1)
    img_norm = img_resized.astype('float32') / 255.0

    # 6. Reshape agar sesuai input model: (1, 128, 128, 1)
    # Tambah dimensi batch (axis 0) dan channel (axis -1)
    img_final = np.expand_dims(img_norm, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)

    return img_final

# --- BAGIAN UTAMA (INPUT USER) ---
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Tombol Prediksi
    if st.button('🔍 Prediksi Ekspresi'):
        with st.spinner('Sedang menganalisis...'):
            try:
                # Proses gambar
                processed_img = preprocess_image(image)

                # Prediksi
                prediction = model.predict(processed_img)
                label_index = np.argmax(prediction)
                confidence_score = prediction[0][label_index]

                # Mapping Label
                classes = {0: 'Happy 😊', 1: 'Sad 😢', 2: 'Angry 😡'}
                predicted_label = classes[label_index]

                # Tampilkan Hasil Utama
                st.markdown("---")
                st.subheader(f"Hasil Prediksi: **{predicted_label}**")
                st.write(f"Confidence (Tingkat Keyakinan): **{confidence_score*100:.2f}%**")

                # Tampilkan Grafik Probabilitas Semua Kelas
                st.write("Detail Probabilitas:")
                probs = prediction[0]
                
                # Membuat progress bar untuk setiap emosi
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("Happy")
                with col2:
                    st.progress(int(probs[0]*100))
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("Sad")
                with col2:
                    st.progress(int(probs[1]*100))
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("Angry")
                with col2:
                    st.progress(int(probs[2]*100))

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")