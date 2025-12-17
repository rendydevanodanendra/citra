import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Emosi", page_icon="😊")

st.title("😊 Klasifikasi Emosi Wajah")
st.write("Upload gambar wajah untuk memprediksi emosi (Happy, Sad, Angry, Netral)")

# 2. Load Model (Cache agar tidak loading ulang terus)
@st.cache_resource
def load_trained_model():
    # Pastikan file model_emosi.h5 ada di satu folder dengan app.py
    model = tf.keras.models.load_model('model_emosi.h5')
    return model

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Definisi Kelas (Sesuai kode training Anda)
class_names = ['Happy', 'Sad', 'Angry', 'Netral']

# 4. Fungsi Preprocessing Gambar
def import_and_predict(image_data, model):
    # --- PENTING: SESUAIKAN BAGIAN INI DENGAN TRAINING ANDA ---
    
    # Contoh: Jika training pakai ukuran 48x48 atau 224x224
    target_size = (48, 48) # Ganti sesuai input shape model Anda (misal 224, 224)
    
    # Resize gambar
    image = ImageOps.fit(image_data, target_size, Image.Resampling.LANCZOS)
    
    # Convert ke array numpy
    img_array = np.asarray(image)
    
    # Normalisasi (jika saat training dibagi 255.0)
    img_array = img_array / 255.0
    
    # Cek apakah model butuh Grayscale (1 channel) atau RGB (3 channel)
    # Jika model Anda grayscale, uncomment baris di bawah ini:
    # if len(img_array.shape) == 3:
    #     img_array = np.mean(img_array, axis=2) # Convert ke grayscale
    
    # Tambah dimensi batch (dari (48,48,3) menjadi (1,48,48,3))
    img_reshape = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_reshape)
    return prediction

# 5. UI untuk Upload File
file = st.file_uploader("Pilih gambar format JPG/PNG", type=["jpg", "png", "jpeg"])

if file is not None:
    # Tampilkan gambar yang diupload user
    image = Image.open(file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    # Tombol Prediksi
    if st.button("Prediksi Emosi"):
        with st.spinner('Sedang memproses...'):
            predictions = import_and_predict(image, model)
            score = tf.nn.softmax(predictions[0])
            
            # Ambil kelas dengan probabilitas tertinggi
            class_idx = np.argmax(predictions, axis=1)[0]
            label = class_names[class_idx]
            confidence = 100 * np.max(score)
            
            st.success(f"Hasil Prediksi: **{label}**")
            st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
            
            # Tampilkan bar chart probabilitas
            st.write("---")
            st.write("Detail Probabilitas:")
            st.bar_chart(
                data=predictions[0], 
            )
            # Mapping index chart manual jika ingin label di sumbu X (opsional)
            for i, name in enumerate(class_names):
                st.write(f"{name}: {predictions[0][i]:.4f}")
