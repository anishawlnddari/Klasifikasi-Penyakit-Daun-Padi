import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# === Load model ===
# Pastikan file model sudah ada di direktori project (misalnya "model.h5")
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# === Daftar kelas penyakit ===
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# === Fungsi prediksi ===
def predict_image(img_rgb):
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds))
    return pred_class, confidence

# === Fungsi proses banyak gambar ===
def proses_gambar(uploaded_files):
    for uploaded_file in uploaded_files[:3]:  # batasi maksimal 3
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error(f"Gagal membaca file: {uploaded_file.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Tampilkan gambar asli
        st.image(img_rgb, caption="Gambar Asli", use_container_width=True)

        # Prediksi
        pred_class, confidence = predict_image(img_rgb)
        st.success(f"Prediksi: **{pred_class}** ({confidence*100:.2f}%)")

# === Fungsi proses kamera ===
def proses_kamera(camera_file):
    if camera_file is not None:
        img = Image.open(camera_file)
        img_rgb = np.array(img)

        st.image(img_rgb, caption="Gambar Kamera", use_container_width=True)

        # Prediksi
        pred_class, confidence = predict_image(img_rgb)
        st.success(f"Prediksi: **{pred_class}** ({confidence*100:.2f}%)")

# === UI Streamlit ===
st.title("🌾 Klasifikasi Penyakit Daun Padi")

menu = st.sidebar.radio("Pilih Input:", ["Upload Gambar", "Kamera Device"])

if menu == "Upload Gambar":
    st.subheader("📂 Upload Gambar Daun (max 3 file)")
    uploaded_files = st.file_uploader(
        "Upload gambar daun", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files:
        proses_gambar(uploaded_files)

elif menu == "Kamera Device":
    st.subheader("📸 Ambil Gambar dari Kamera")
    camera_file = st.camera_input("Ambil gambar daun padi")
    if camera_file:
        proses_kamera(camera_file)
