import cv2
import numpy as np
from sklearn.cluster import KMeans

# === Resize ===
def resize_image(img_bgr, target_size=(224, 224)):
    resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    return resized

# === Segmentasi HSV ===
lower_yellow = np.array([15, 40, 40])
upper_yellow = np.array([35, 255, 255])
lower_brown  = np.array([0, 30, 20])
upper_brown  = np.array([20, 255, 200])

def segmentasi_penyakit(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Mask penyakit
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_brown  = cv2.inRange(img_hsv, lower_brown, upper_brown)
    mask_penyakit = cv2.bitwise_or(mask_yellow, mask_brown)

    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    mask_penyakit = cv2.morphologyEx(mask_penyakit, cv2.MORPH_OPEN, kernel)
    mask_penyakit = cv2.morphologyEx(mask_penyakit, cv2.MORPH_CLOSE, kernel)

    # Background putih
    background_putih = np.full_like(img_bgr, 255)

    # Gabungkan
    hasil = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_penyakit)
    latar = cv2.bitwise_and(background_putih, background_putih, mask=cv2.bitwise_not(mask_penyakit))
    hasil_final = cv2.add(hasil, latar)

    return mask_penyakit, hasil_final

# === KMeans Filtering Hijau ===
def remove_green_kmeans(img_rgb, k=3):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Cari cluster hijau
    green_cluster = None
    for i, center in enumerate(centers):
        h_val = center[0]  # Hue
        if 35 <= h_val <= 85:  # hijau
            green_cluster = i
            break

    mask = (labels != green_cluster).astype(np.uint8).reshape(h, w)

    result = img_rgb.copy()
    result[mask == 0] = [0, 0, 0]  # hijau → hitam

    return result, mask
