#app.py resize

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time  # untuk logging waktu
from PIL import Image

# --- konfigurasi halaman ---
st.set_page_config(page_title="Klasifikasi Penyakit Daun Padi", layout="wide")

# --- import fungsi ---
from preprocessing import resize_image

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

@st.cache_resource
def load_model_once():
    return tf.keras.models.load_model("src/model/best_model_finetune_resize_model.h5")

model = load_model_once()

st.title("Klasifikasi Penyakit Daun Padi")

tab1, tab2 = st.tabs(["📂 Upload Gambar", "📷 Kamera"])

# --- fungsi utama ---
def proses_gambar(files_to_process):
    for idx, uploaded_file in enumerate(files_to_process, start=1):
        st.markdown(f"## 🖼️ Gambar {idx}")

        # Baca file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Layout kiri-kanan
        col1, col2 = st.columns([1, 2])  # kiri lebih kecil, kanan lebih besar

        with col1:
            st.image(img_rgb, caption="Gambar Asli", width=200)

        with col2:
            # Mulai hitung waktu
            start_time = time.time()

            # 1) Resize (224x224)
            resized_bgr = resize_image(img_bgr, (224, 224))
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

            # 2) Prediksi
            input_tensor = np.expand_dims(resized_rgb, axis=0) / 255.0
            preds = model.predict(input_tensor, verbose=0)[0]

            elapsed_time = time.time() - start_time

            pred_class = CLASS_NAMES[np.argmax(preds)]
            pred_conf = np.max(preds) * 100

            st.subheader("📌 Hasil Prediksi")
            st.success(f"Penyakit Terdeteksi: **{pred_class}** ({pred_conf:.2f}%)")

            # Logging waktu
            st.info(f"⚡ Kecepatan respons prediksi: {elapsed_time:.3f} detik")

            # Tabel probabilitas
            prob_df = pd.DataFrame({
                "Penyakit": CLASS_NAMES,
                "Probabilitas (%)": (preds * 100).round(2)
            }).sort_values(by="Probabilitas (%)", ascending=False)

            st.subheader("📊 Probabilitas Semua Kelas")
            st.dataframe(prob_df, use_container_width=True)

            # Bar chart
            st.subheader("📈 Visualisasi Probabilitas")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(CLASS_NAMES, preds * 100, color="teal")
            ax.set_xlabel("Probabilitas (%)")
            ax.set_title("Prediksi Probabilitas per Kelas")
            ax.invert_yaxis()
            st.pyplot(fig)

        st.markdown("---")

# --- Tab 1: Upload ---
with tab1:
    uploaded_files = st.file_uploader(
        "Upload hingga 3 gambar daun padi (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        proses_gambar(uploaded_files[:3])  # maks 3

# --- Tab 2: Kamera ---
with tab2:
    camera_file = st.camera_input("Ambil foto daun padi")
    if camera_file:
        proses_gambar([camera_file])
