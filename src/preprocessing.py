#preprocessing.py resize

import cv2

def resize_image(img_bgr, target_size=(224, 224)):
    """
    Resize BGR image ke target_size (default 224×224).
    """
    return cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
