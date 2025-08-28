import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Import fungsi dari preprocessing.py
from preprocessing import resize_image, segmentasi_penyakit, remove_green_kmeans

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

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("src/model/best_model_finetune.h5")  
    return model

model = load_model()

st.title("🌾 Klasifikasi Penyakit Daun Padi")

# === Tab Menu ===
tab1, tab2 = st.tabs(["📂 Upload Gambar", "📷 Kamera"])

import pandas as pd
import matplotlib.pyplot as plt

def proses_gambar(files_to_process):
    for idx, uploaded_file in enumerate(files_to_process, start=1):
        st.markdown(f"## 🖼️ Gambar {idx}")

        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Tampilkan gambar asli
        st.image(img_rgb, caption="Gambar Asli", use_column_width=True)   # <-- diperbaiki

        # Step 1: Resize
        resized_bgr = resize_image(img_bgr, (224, 224))

        # Step 2: Segmentasi HSV
        mask_hsv, seg_hsv = segmentasi_penyakit(resized_bgr)

        # Step 3: KMeans Filtering Hijau
        img_rgb_seg = cv2.cvtColor(seg_hsv, cv2.COLOR_BGR2RGB)
        kmeans_result, kmeans_mask = remove_green_kmeans(img_rgb_seg, k=3)

        # Prediksi
        input_tensor = np.expand_dims(kmeans_result, axis=0) / 255.0
        preds = model.predict(input_tensor)[0]

        pred_class = CLASS_NAMES[np.argmax(preds)]
        pred_conf = np.max(preds) * 100

        # Output hasil
        st.subheader("📌 Hasil Prediksi")
        st.success(f"Penyakit Terdeteksi: **{pred_class}** ({pred_conf:.2f}%)")

        # Tabel probabilitas
        prob_df = pd.DataFrame({
            "Penyakit": CLASS_NAMES,
            "Probabilitas (%)": (preds * 100).round(2)
        }).sort_values(by="Probabilitas (%)", ascending=False)

        st.subheader("📊 Probabilitas Semua Kelas")
        st.dataframe(prob_df, use_container_width=True)

        # Visualisasi bar chart
        st.subheader("📈 Visualisasi Probabilitas")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(CLASS_NAMES, preds * 100)
        ax.set_xlabel("Probabilitas (%)")
        ax.set_title("Prediksi Probabilitas per Kelas")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.markdown("---")

# === Tab 1: Upload Gambar ===
with tab1:
    uploaded_files = st.file_uploader(
        "Upload hingga 3 gambar daun (jpg/jpeg/png)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        proses_gambar(uploaded_files[:3])  # batasi maksimal 3

# === Tab 2: Kamera ===
with tab2:
    camera_file = st.camera_input("Ambil gambar dengan kamera")
    if camera_file:
        proses_gambar([camera_file])
