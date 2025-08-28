import cv2
import numpy as np
from sklearn.cluster import KMeans

# ---------- parameter warna ----------
lower_yellow = np.array([15,  40,  40])
upper_yellow = np.array([35, 255, 255])
lower_brown  = np.array([ 0,  30,  20])
upper_brown  = np.array([20, 255, 200])

def resize_image(img_bgr, target_size=(224, 224)):
    """Resize BGR image ke target_size (default 224×224)."""
    return cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

def segmentasi_penyakit(img_bgr):
    """
    Segmentasi daerah penyakit (kuning/cokelat) dengan HSV.
    Returns:
        mask_penyakit : mask 1-channel (0/255)
        hasil_final   : BGR image dengan background putih
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_brown  = cv2.inRange(img_hsv, lower_brown,  upper_brown)
    mask_penyakit = cv2.bitwise_or(mask_yellow, mask_brown)

    kernel = np.ones((5, 5), np.uint8)
    mask_penyakit = cv2.morphologyEx(mask_penyakit, cv2.MORPH_OPEN, kernel)
    mask_penyakit = cv2.morphologyEx(mask_penyakit, cv2.MORPH_CLOSE, kernel)

    background = np.full_like(img_bgr, 255)
    hasil = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_penyakit)
    latar = cv2.bitwise_and(background, background,
                            mask=cv2.bitwise_not(mask_penyakit))
    hasil_final = cv2.add(hasil, latar)

    return mask_penyakit, hasil_final


def remove_green_kmeans(img_rgb, k=3):
    """
    Hilangkan area hijau dengan KMeans pada channel Hue.
    Returns:
        filtered : RGB image dengan area hijau di-black-kan
        mask     : mask 1-channel (0/255) area NON-hijau
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    green_cluster = None
    for i, center in enumerate(centers):
        if 35 <= center[0] <= 85:
            green_cluster = i
            break

    mask = (labels != green_cluster).astype(np.uint8).reshape(h, w)
    filtered = img_rgb.copy()
    filtered[mask == 0] = [0, 0, 0]

    return filtered, mask