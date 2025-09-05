import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Klasifikasi Penyakit Daun Padi", layout="wide")

# --- import fungsi dari preprocessing.py ---
from preprocessing import resize_image, segmentasi_penyakit, remove_green_kmeans

# --- Daftar kelas ---
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

# --- Load model hanya sekali ---
@st.cache_resource
def load_model_once():
    return tf.keras.models.load_model("src/model/best_model_finetune.h5")

model = load_model_once()

st.title("🌾 Klasifikasi Penyakit Daun Padi")

# --- Tab menu ---
tab1, tab2 = st.tabs(["📂 Upload Gambar", "📷 Kamera"])


# --- Fungsi utama ---
def proses_gambar(files_to_process):
    for idx, uploaded_file in enumerate(files_to_process, start=1):
        st.markdown(f"## 🖼️ Gambar {idx}")

        # Baca file gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Layout kiri-kanan
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(img_rgb, caption="Gambar Asli", width=200)

        with col2:
            # Mulai timer
            start_time = time.time()

            # 1) Resize
            resized_bgr = resize_image(img_bgr, (224, 224))

            # 2) Segmentasi HSV -> (mask, hasil)
            mask_hsv, seg_hsv = segmentasi_penyakit(resized_bgr)

            # 3) KMeans filter hijau -> (filtered, mask)
            img_rgb_seg = cv2.cvtColor(seg_hsv, cv2.COLOR_BGR2RGB)
            kmeans_result, _ = remove_green_kmeans(img_rgb_seg, k=3)

            # 4) Prediksi
            input_tensor = np.expand_dims(kmeans_result, axis=0) / 255.0
            preds = model.predict(input_tensor)[0]

            # Selesai timer
            elapsed_time = time.time() - start_time

            # Ambil hasil prediksi
            pred_class = CLASS_NAMES[np.argmax(preds)]
            pred_conf = np.max(preds) * 100

            # Tampilkan hasil
            st.subheader("📌 Hasil Prediksi")
            st.success(f"Penyakit Terdeteksi: **{pred_class}** ({pred_conf:.2f}%)")

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
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(CLASS_NAMES, preds * 100, color="teal")
            ax.set_xlabel("Probabilitas (%)")
            ax.set_title("Prediksi Probabilitas per Kelas")
            ax.invert_yaxis()
            st.pyplot(fig)

        st.markdown("---")


# --- Tab 1: Upload Gambar ---
with tab1:
    uploaded_files = st.file_uploader(
        "Upload hingga 3 gambar daun padi (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        proses_gambar(uploaded_files[:3])  # batasi max 3 gambar

# --- Tab 2: Kamera ---
with tab2:
    camera_file = st.camera_input("Ambil foto daun padi")
    if camera_file:
        proses_gambar([camera_file])
