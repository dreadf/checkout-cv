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


def empty_table_reference(frames):
    """Rata-ratakan beberapa frame meja kosong jadi satu gambar referensi, dipakai saat tombol K ditekan."""
    return np.mean(np.stack(frames), axis=0).astype(np.uint8)


def find_object_blobs(frame, background):
    """Sama seperti find_object_boxes, tapi hasilnya lengkap dengan mask bentuk asli benda (bukan
    hanya kotak pembungkus), supaya cek benda asing tidak salah hitung lantai kosong di antara barang
    yang berdekatan. Hasilnya list dict {"box": (x1, y1, x2, y2), "area", "mask"}, dari yang terbesar.
    """
    mask = clean_mask(difference_mask(frame, background))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area >= config.MIN_LIVE_BLOB_AREA and min(w, h) >= config.MIN_BLOB_SIDE:
            blob_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
            blobs.append({"box": (x, y, x + w, y + h), "area": area, "mask": blob_mask})
    return sorted(blobs, key=lambda b: b["area"], reverse=True)


def _covered_ratio(mask, known_boxes):
    """Berapa persen piksel bentuk benda yang jatuh di dalam salah satu kotak produk yang dikenali.
    Dihitung dari piksel bentuk asli, bukan kotak pembungkusnya, supaya lantai kosong di antara
    barang yang berdekatan tidak ikut dianggap "belum dijelaskan"."""
    object_pixels = mask > 0
    total = int(object_pixels.sum())
    if total == 0:
        return 1.0
    covered = np.zeros_like(object_pixels)
    for x1, y1, x2, y2 in known_boxes:
        covered[y1:y2, x1:x2] = True
    return (object_pixels & covered).sum() / total


def uncovered_blobs(blobs, known_boxes):
    """Benda yang sebagian besar areanya tidak tertutup kotak produk dikenal."""
    return [b for b in blobs if _covered_ratio(b["mask"], known_boxes) < config.OBJECT_COVER_MIN]


def _is_near_edge(blob, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = blob["box"]
    return x1 < config.EDGE_MARGIN or y1 < config.EDGE_MARGIN or x2 > width - config.EDGE_MARGIN or y2 > height - config.EDGE_MARGIN


def iou(a, b):
    """Seberapa besar dua kotak (x1, y1, x2, y2) saling menimpa: luas irisan dibagi luas gabungan.
    0 berarti tidak bersentuhan, 1 berarti kotaknya identik."""
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def remove_overlaps(items):
    """Buang kotak kembar: kotak yang IoU-nya dengan kotak lebih kuat di atas OVERLAP_IOU dianggap benda
    yang sama. Tidak peduli kelasnya, jadi satu botol yang ditebak dua nama tinggal satu.

    items: list dict dengan "box" dan "conf", dan boleh "lost" (frame sejak terakhir terlihat).
    Yang paling baru terlihat didahulukan, lalu yang confidence-nya tertinggi.
    """
    kept = []
    for item in sorted(items, key=lambda i: (i.get("lost", 0), -i["conf"])):
        if all(iou(item["box"], k["box"]) <= config.OVERLAP_IOU for k in kept):
            kept.append(item)
    return kept


def is_moving(frame, previous_frame):
    """Bagian piksel yang berubah dari frame sebelumnya kalau di atas MOTION_MIN, selain itu 0.
    Angkanya ikut ditampilkan ke pembeli, jadi gampang dibandingkan dengan MOTION_MIN saat menyetel."""
    if previous_frame is None or frame.shape != previous_frame.shape:
        return 0.0
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    changed = float((diff > config.MOTION_PIXEL_DIFF).mean())
    return changed if changed > config.MOTION_MIN else 0.0


def table_conditions(frame, previous_frame, blobs, known_boxes):
    """Gabungkan cek benda asing, tepi, dan gerak jadi satu dictionary kondisi untuk app/lock.py.

    "tidak_dikenal" adalah jumlah benda di meja yang tidak tertutup kotak produk dikenal (tangan, HP,
    dompet, atau produk yang tidak terdeteksi YOLO). "tepi" adalah jumlah di antaranya yang dekat
    pinggir area meja. Keduanya saling lepas supaya pesan ke pembeli spesifik. Benda yang sudah dikenal
    YOLO tapi kotaknya menyentuh pinggir frame tidak dianggap masalah, karena TABLE_REGION dipotong pas
    dengan area taruh barang. Nilai 0 berarti tidak ada masalah.
    """
    uncovered = uncovered_blobs(blobs, known_boxes)
    near_edge = [b for b in uncovered if _is_near_edge(b, frame.shape)]
    return {
        "tepi": len(near_edge),
        "tidak_dikenal": len(uncovered) - len(near_edge),
        "bergerak": is_moving(frame, previous_frame),
    }
