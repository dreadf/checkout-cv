"""Pilih area meja dengan menarik kotak di layar. Hasilnya dicetak, salin ke TABLE_REGION di config.py.

Cara pakai dari folder utama repo:
    python tools/set_table_region.py
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from app import camera


def main():
    cap = camera.open_camera()
    if cap is None:
        print("Kamera tidak bisa dibuka. Cek CAMERA_INDEX di config.py dan izin kamera di sistem.")
        return
    # Buang beberapa frame awal karena kamera butuh waktu menyesuaikan cahaya.
    for _ in range(config.CAPTURE_WARMUP_FRAMES):
        ok, frame = cap.read()
    cap.release()

    print("Tarik kotak mengikuti pinggir alas meja, lalu tekan Enter. Tekan C untuk batal.")
    x, y, w, h = cv2.selectROI(config.WINDOW_TITLE, frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("Dibatalkan.")
        return
    print("Salin baris ini ke config.py:")
    print(f"TABLE_REGION = {(int(x), int(y), int(w), int(h))}")


if __name__ == "__main__":
    main()
