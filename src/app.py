import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Import fungsi preprocessing
from preprocessing import resize_image

# === Load Model ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("src/model/best_model_finetune.h5")

model = load_model()

# === Kelas penyakit ===
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# === Fungsi Prediksi ===
def predict_image(img_rgb):
    img_resized = resize_image(img_rgb, (224,224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return CLASS_NAMES[class_idx], confidence

# === Fungsi Proses Gambar ===
def proses_gambar(uploaded_files):
    for uploaded_file in uploaded_files[:3]:  # batasi maksimal 3
        try:
            # Baca gambar dengan PIL
            img = Image.open(uploaded_file).convert("RGB")
            img_rgb = np.array(img)

            # Tampilkan gambar asli
            st.image(img_rgb, caption="Gambar Asli", use_container_width=True)

            # Prediksi
            pred_class, confidence = predict_image(img_rgb)
            st.success(f"Prediksi: **{pred_class}** ({confidence*100:.2f}%)")

        except Exception as e:
            st.error(f"Gagal memproses file {uploaded_file.name}: {e}")

# === UI Streamlit ===
st.title("🌾 Klasifikasi Penyakit Daun Padi")

menu = st.sidebar.radio("Pilih Input Gambar", ["📤 Upload Gambar", "📸 Kamera"])

if menu == "📤 Upload Gambar":
    uploaded_files = st.file_uploader(
        "Upload hingga 3 gambar daun",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        proses_gambar(uploaded_files)

elif menu == "📸 Kamera":
    camera_image = st.camera_input("Ambil foto daun dengan kamera")
    if camera_image is not None:
        proses_gambar([camera_image])
