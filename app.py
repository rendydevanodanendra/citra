import streamlit as st
import cv2
import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon=" ",
    layout="centered"
)

# ---------------------------------------------------------------------
# FUNGSI PREPROCESSING (HARUS SAMA PERSIS DENGAN TRAINING)
# ---------------------------------------------------------------------
def preprocess_image(image):
    # 1. Konversi dari PIL Image (RGB) ke Numpy Array (BGR/RGB)
    img_array = np.array(image)

    # 2. Konversi ke Grayscale (jika gambar berwarna)
    # Cek apakah gambar punya 3 channel (RGB)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # 3. Histogram Equalization (PENTING: Sesuai training)
    img_eq = cv2.equalizeHist(img_gray)

    # 4. Resize ke 128x128
    img_resized = cv2.resize(img_eq, (128, 128))

    # 5. Normalisasi (0-1)
    img_norm = img_resized.astype('float32') / 255.0

    # 6. Reshape agar sesuai input model: (1, 128, 128, 1)
    # Tambah dimensi batch dan channel
    img_final = np.expand_dims(img_norm, axis=0) # Jadi (1, 128, 128)
    img_final = np.expand_dims(img_final, axis=-1) # Jadi (1, 128, 128, 1)

    return img_final

# ---------------------------------------------------------------------
# LOAD MODEL (CACHE SUPAYA LEBIH CEPAT)
# ---------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    # Pastikan file .h5 ada satu folder dengan app.py
    model = load_model('model_emosi.h5')
    return model

# Load model saat aplikasi mulai
try:
    model = load_trained_model()
    model_status = "Model berhasil dimuat!"
except Exception as e:
    model_status = f"Error memuat model: {e}"

# ---------------------------------------------------------------------
# TAMPILAN UTAMA (UI)
# ---------------------------------------------------------------------
st.title("Facial Expression Recognition")
st.write("Klasifikasi Ekspresi Wajah: **Happy, Sad, Angry, Netral**")
st.write("Menggunakan Arsitektur **MobileNetV2**")

# Tampilkan status model (opsional, untuk debug)
# st.text(model_status)

uploaded_file = st.file_uploader("Upload gambar wajah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Tombol Prediksi
    if st.button('Prediksi Ekspresi'):
        with st.spinner('Sedang memproses...'):
            try:
                # Preprocessing
                processed_img = preprocess_image(image)
                
                # Prediksi
                prediction = model.predict(processed_img)
                label_index = np.argmax(prediction)
                confidence = prediction[0][label_index]

                # Mapping Label
                classes = {0: 'Happy 😊', 1: 'Sad 😢', 2: 'Angry 😡', 3: 'Netral 😐'}
                result_label = classes[label_index]

                # Tampilkan Hasil
                st.success(f"Prediksi: **{result_label}**")
                st.info(f"Tingkat Kepercayaan (Confidence): **{confidence*100:.2f}%**")

                # Tampilkan Grafik Probabilitas
                st.write("---")
                st.write("Probabilitas per Kelas:")
                prob_data = {
                    'Ekspresi': ['Happy', 'Sad', 'Angry', 'Netral'],
                    'Probabilitas': prediction[0]
                }
                st.bar_chart(prob_data, x='Ekspresi', y='Probabilitas')

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# Footer
st.markdown("---")
st.caption("Dikembangkan oleh Kelompok 9 - Universitas Trunojoyo Madura")


