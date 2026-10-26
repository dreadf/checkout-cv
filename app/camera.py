import sys

import cv2

import config


def open_camera():
    """Buka webcam. Hasilnya None kalau kamera tidak bisa dibuka."""
    # Di Windows, backend DirectShow lebih cepat terbuka daripada bawaan.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def crop_to_table(frame):
    """Potong frame ke area meja. Kalau area belum diatur, pakai seluruh frame."""
    if config.TABLE_REGION is None:
        return frame
    x, y, w, h = config.TABLE_REGION
    return frame[y:y + h, x:x + w]


def resize_to_width(frame):
    """Kecilkan frame ke lebar PROCESS_WIDTH dengan rasio yang sama."""
    height, width = frame.shape[:2]
    new_height = round(height * config.PROCESS_WIDTH / width)
    return cv2.resize(frame, (config.PROCESS_WIDTH, new_height))


def preprocess(frame):
    """Crop lalu resize. Foto training dan frame live harus lewat fungsi ini."""
    return resize_to_width(crop_to_table(frame))


def camera_problem(frame, previous_frame):
    """'gelap', 'terhalang', 'membeku', atau None kalau kamera sehat.

    Macet dicek dengan satu kali sama persis piksel per piksel, bukan beberapa frame berturut-turut,
    karena sensor kamera asli selalu punya sedikit noise, jadi frame yang identik sudah cukup aneh.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 2].mean() < config.CAMERA_DARK_V:
        return "gelap"
    if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std() < config.CAMERA_BLOCKED_STD:
        return "terhalang"
    if previous_frame is not None and frame.shape == previous_frame.shape and (frame == previous_frame).all():
        return "membeku"
    return None
