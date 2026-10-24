"""Buat label kotak otomatis untuk foto produk di data/product_photos/.

Cara pakai dari folder utama repo:
    python tools/auto_label.py

Cara kerjanya: foto dibandingkan ke gambar meja kosong, lalu benda yang berbeda diberi kotak.
Nama kelas diambil dari nama folder. Meja kosongnya dibuat dari median banyak foto produk.
Barang selalu berpindah posisi antar foto, jadi nilai tengah tiap piksel adalah lantai.

Label yang sudah ada ditimpa. Foto yang tidak ditemukan bendanya, atau ada lebih dari satu benda,
tidak diberi label.
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from app import checks
from app.products import load_products

PHOTOS_DIR = config.DATA_DIR / "product_photos"


def refine_background(frames, background):
    """Hitung median lagi tanpa piksel benda, supaya bayangan samar barang hilang dari meja kosong."""
    kernel = np.ones((config.BACKGROUND_EXCLUDE_KERNEL,) * 2, np.uint8)
    masks = [cv2.dilate(checks.clean_mask(checks.difference_mask(frame, background)), kernel) > 0
             for frame in frames]
    stack = np.stack(frames).astype(np.float32)
    stack[np.stack(masks)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        refined = np.nanmedian(stack, axis=0)
    missing = np.isnan(refined)
    refined[missing] = np.broadcast_to(background, refined.shape)[missing]
    return refined.astype(np.uint8)


def build_background(photos):
    """Buat satu gambar meja kosong dari median foto yang dipilih merata dari semua foto."""
    step = max(1, len(photos) // config.BACKGROUND_MAX_FRAMES)
    frames = [cv2.imread(str(path)) for path in photos[::step][:config.BACKGROUND_MAX_FRAMES]]
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return refine_background(frames, median)


def pick_object(blobs):
    """Pilih benda terbesar. Hasilnya None kalau tidak ada benda, atau ada benda kedua yang hampir sama besar."""
    if not blobs:
        return None
    if len(blobs) > 1 and blobs[1][4] >= config.AMBIGUOUS_RATIO * blobs[0][4]:
        return None
    return blobs[0]


def yolo_line(class_id, blob, image_shape):
    """Satu baris label YOLO: kelas, titik tengah x dan y, lebar, tinggi. Semua relatif 0 sampai 1."""
    height, width = image_shape[:2]
    x, y, w, h, _ = blob
    return f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}\n"


def main():
    class_ids = {name: i for i, name in enumerate(load_products(config.PRODUCTS_CSV))}
    photos = [photo for folder in sorted(PHOTOS_DIR.iterdir()) if folder.is_dir() for photo in sorted(folder.glob("*.jpg"))]
    print(f"Membuat gambar meja kosong dari {len(photos)} foto...")
    background = build_background(photos)

    labeled = 0
    for photo in photos:
        image = cv2.imread(str(photo))
        blob = pick_object(checks.find_object_boxes(image, background))
        label_path = photo.with_suffix(".txt")
        if blob is None:
            label_path.unlink(missing_ok=True)
            continue
        label_path.write_text(yolo_line(class_ids[photo.parent.name], blob, image.shape), encoding="utf-8")
        labeled += 1
    print(f"Selesai. {labeled} dari {len(photos)} foto berlabel.")


if __name__ == "__main__":
    main()
