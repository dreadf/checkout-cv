import cv2
import numpy as np

import config


def gradient_strength(image):
    """Kekuatan tepi tiap piksel. Botol bening punya banyak tepi dari pantulan, meja polos hampir tidak."""
    blurred = cv2.GaussianBlur(image, (config.GRADIENT_BLUR_KERNEL,) * 2, 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def difference_mask(frame, background):
    """Tandai piksel yang berbeda dari meja kosong, lewat beda warna atau beda tepi."""
    kernel = (config.OBJECT_BLUR_KERNEL,) * 2
    lab_frame = cv2.cvtColor(cv2.GaussianBlur(frame, kernel, 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_back = cv2.cvtColor(cv2.GaussianBlur(background, kernel, 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab_frame - lab_back
    # Kamera sering mengubah kecerahan otomatis, jadi pergeseran rata-rata dibuang dulu.
    delta -= np.median(delta.reshape(-1, 3), axis=0)
    delta[:, :, 0] *= config.OBJECT_L_WEIGHT
    color_diff = np.sqrt((delta ** 2).sum(axis=2)) > config.OBJECT_CHROMA_THRESHOLD
    edge_diff = np.abs(gradient_strength(frame) - gradient_strength(background)) > config.OBJECT_GRADIENT_THRESHOLD
    return (color_diff | edge_diff).astype(np.uint8) * 255


def clean_mask(mask):
    """Daerah abaian dikosongkan, closing menyatukan pecahan satu benda, opening membuang garis tipis."""
    for x, y, w, h in config.OBJECT_IGNORE_REGIONS:
        mask[y:y + h, x:x + w] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((config.OBJECT_CLOSE_KERNEL,) * 2, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((config.OBJECT_OPEN_KERNEL,) * 2, np.uint8))


def find_object_boxes(frame, background):
    """Cari benda di meja. Hasilnya list (x, y, w, h, area), dari yang terbesar."""
    mask = clean_mask(difference_mask(frame, background))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area >= config.MIN_BLOB_AREA and min(w, h) >= config.MIN_BLOB_SIDE:
            blobs.append((x, y, w, h, area))
    return sorted(blobs, key=lambda blob: blob[4], reverse=True)
