import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from PIL import Image

# ====================
# Load Model
# ====================
MODEL_PATH = "src/model/best_model_finetune.h5"
model = load_model(MODEL_PATH)

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

# ====================
# Preprocessing
# ====================

def resize_image(img_bgr, target_size=(224, 224)):
    return cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

# --- HSV Segmentasi ---
lower_yellow = np.array([15, 40, 40])
upper_yellow = np.array([35, 255, 255])
lower_brown  = np.array([0, 30, 20])
upper_brown  = np.array([20, 255, 200])

def segmentasi_penyakit(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_brown  = cv2.inRange(img_hsv, lower_brown, upper_brown)
    mask_penyakit = cv2.bitwise_or(mask_yellow, mask_brown)

    kernel = np.ones((5,5), np.uint8)
    mask_penyakit = cv2.morphologyEx(mask_penyakit, cv2.MORPH_OPEN, kernel)
    mask_penyakit = cv2.morphologyEx(mask_penyakit, cv2.MORPH_CLOSE, kernel)

    background_putih = np.full_like(img_bgr, 255)
    hasil = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_penyakit)
    latar = cv2.bitwise_and(background_putih, background_putih, mask=cv2.bitwise_not(mask_penyakit))
    hasil_final = cv2.add(hasil, latar)

    return mask_penyakit, hasil_final

# --- KMeans Filtering Hijau ---
def remove_green_kmeans(img_rgb, k=3):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    green_cluster = None
    for i, center in enumerate(centers):
        h_val = center[0]
        if 35 <= h_val <= 85:  # range hijau
            green_cluster = i
            break

    mask = (labels != green_cluster).astype(np.uint8).reshape(h, w)
    result = img_rgb.copy()
    result[mask == 0] = [0, 0, 0]

    return result, mask

# ====================
# Prediksi
# ====================
def predict_image(img_rgb):
    mask, hasil_segmentasi = segmentasi_penyakit(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    filtered, _ = remove_green_kmeans(img_rgb)

    img_resized = resize_image(filtered, (224,224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_idx], confidence, hasil_segmentasi, filtered, img_resized

# ====================
# Streamlit UI
# ====================
st.set_page_config(page_title="Klasifikasi Penyakit Daun Padi", layout="wide")
st.title("🌾 Klasifikasi Penyakit Daun Padi")

menu = st.sidebar.radio("Pilih Input:", ["📤 Upload Gambar", "📸 Kamera"])

def proses_gambar(images):
    """
    Menerima list objek file gambar (dari st.file_uploader atau st.camera_input),
    lalu menampilkan hasil prediksi dan visualisasi langkah-langkah preprocessing.
    """
    for img in images:
        # Buka dan konversi ke RGB
        img_rgb = np.array(Image.open(img).convert("RGB"))

        # Jalankan pipeline prediksi
        label, confidence, hasil_segmentasi, filtered, img_resized = predict_image(img_rgb)

        # Tampilkan hasil
        st.subheader(f"📌 Hasil Prediksi: **{label}**")
        st.write(f"Confidence: {confidence:.2%}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Gambar Asli", use_column_width=True)
            st.image(hasil_segmentasi, caption="Segmentasi HSV", use_column_width=True)
        with col2:
            st.image(filtered, caption="Filtering Hijau (KMeans)", use_column_width=True)
            st.image(img_resized, caption="Input Model (224×224)", use_column_width=True)

        st.markdown("---")

if menu == "📤 Upload Gambar":
    uploaded_files = st.file_uploader("Upload hingga 3 gambar daun padi", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("⚠️ Maksimal 3 gambar saja.")
        else:
            proses_gambar(uploaded_files)

elif menu == "📸 Kamera":
    cam_file = st.camera_input("Ambil foto daun padi")
    if cam_file is not None:
        proses_gambar([cam_file])
