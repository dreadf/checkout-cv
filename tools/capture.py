"""Ambil foto dengan kamera dan area meja yang sama seperti aplikasi.

Contoh dari folder utama repo:
    python tools/capture.py --mode product --class aqua_600ml
    python tools/capture.py --mode background
    python tools/capture.py --mode scene

Spasi mengambil satu foto, B menyalakan atau mematikan burst (satu foto setiap beberapa detik), Q selesai.
Setiap kali dijalankan terbentuk satu sesi. ID sesinya (tanggal dan jam) ada di nama file
dan dipakai untuk membagi data train dan val.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from app import camera
from app.products import load_products

FONT = cv2.FONT_HERSHEY_SIMPLEX


def capture_loop(cap, folder, session, prefix, title):
    """Tampilkan kamera dan simpan foto ke folder. Hasilnya jumlah foto yang tersimpan di sesi ini."""
    folder.mkdir(parents=True, exist_ok=True)
    count = len(list(folder.glob(f"{session}_{prefix}*.jpg")))
    burst, last_shot = False, 0.0
    while True:
        ok, raw = cap.read()
        if not ok:
            print("Gagal membaca gambar dari kamera.")
            return count
        frame = camera.preprocess(raw)

        view = frame.copy()
        status = f"{title}   Foto sesi ini: {count}   Burst: {'AKTIF' if burst else 'mati'}"
        cv2.putText(view, status, (10, 25), FONT, config.FONT_SCALE, config.COLOR_WARN, 2)
        cv2.putText(view, "Spasi: foto   B: burst   Q: selesai", (10, 50), FONT, config.FONT_SCALE, config.COLOR_WARN, 2)
        cv2.imshow(config.WINDOW_TITLE, view)

        key = cv2.waitKey(1) & 0xFF
        key = chr(key).lower() if key != 255 else ""
        now = time.perf_counter()
        take = key == " " or (burst and now - last_shot >= config.BURST_INTERVAL)
        if key == "b":
            burst, last_shot = not burst, now
        elif key == "q":
            return count
        if take:
            count += 1
            path = folder / f"{session}_{prefix}{count:03d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            last_shot = now


def main():
    parser = argparse.ArgumentParser(description="Ambil foto untuk dataset.")
    parser.add_argument("--mode", required=True, choices=("product", "background", "scene"))
    parser.add_argument("--class", dest="class_name", help="Nama kelas, wajib untuk mode product")
    args = parser.parse_args()

    products = load_products(config.PRODUCTS_CSV)
    if args.mode == "product" and args.class_name not in products:
        print("Pakai --class dengan salah satu nama ini: " + ", ".join(products))
        return
    folder, prefix = {
        "product": (config.DATA_DIR / "product_photos" / str(args.class_name), ""),
        "background": (config.DATA_DIR / "background_photos", "bg"),
        "scene": (config.DATA_DIR / "scene_photos_train", "scene"),
    }[args.mode]

    cap = camera.open_camera()
    if cap is None:
        print("Kamera tidak bisa dibuka. Cek CAMERA_INDEX di config.py dan izin kamera di sistem.")
        return
    for _ in range(config.CAPTURE_WARMUP_FRAMES):
        cap.read()

    session = datetime.now().strftime("%Y%m%d_%H%M")
    count = capture_loop(cap, folder, session, prefix, args.mode)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. {count} foto tersimpan di sesi {session}.")


if __name__ == "__main__":
    main()
