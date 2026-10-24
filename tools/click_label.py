"""Beri label foto adegan dengan klik, tanpa menggambar kotak.

Cara pakai dari folder utama repo:
    python tools/click_label.py

Untuk setiap barang di foto: tekan angka produknya (1 sampai 8), lalu klik di tengah barang itu.
FastSAM menentukan kotak barang dari titik klik. Angka produk tetap terpilih sampai kamu menekan
angka lain, jadi untuk dua barang yang sama cukup klik dua kali. Kalau kotak otomatisnya salah,
tarik mouse untuk menggambar kotak sendiri.

Tombol: Z batalkan kotak terakhir, N atau Spasi simpan lalu lanjut, X pindahkan foto ke folder dibuang,
Q simpan lalu keluar. Hanya foto yang belum punya label yang ditampilkan.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from app.products import load_products

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCENES_DIR = config.DATA_DIR / "scene_photos_train"


def mask_box(model, image, x, y):
    """Kotak (x1, y1, x2, y2) dari benda di titik (x, y). Hasilnya None kalau kliknya meleset."""
    result = model(image, device="cpu", retina_masks=True, imgsz=config.SAM_IMG_SIZE,
                   points=[[x, y]], labels=[1], verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    mask = result.masks.data[0].cpu().numpy() > 0.5
    if mask.mean() > config.CLICK_MAX_AREA:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def write_labels(photo, boxes, image_shape):
    """Tulis kotak (class_id, x1, y1, x2, y2) jadi file label YOLO di samping foto."""
    height, width = image_shape[:2]
    lines = [f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
             f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n" for class_id, x1, y1, x2, y2 in boxes]
    photo.with_suffix(".txt").write_text("".join(lines), encoding="utf-8")


def draw_view(image, boxes, product_names, current, message):
    """Gambar foto diperbesar dengan kotak, lalu panel berisi nomor produk dan petunjuk di kanan."""
    scale = config.CLICK_SCALE
    view = cv2.resize(image, None, fx=scale, fy=scale)
    for class_id, x1, y1, x2, y2 in boxes:
        color = config.CLASS_COLORS[class_id % len(config.CLASS_COLORS)]
        cv2.rectangle(view, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), color, config.BOX_THICKNESS)
        cv2.putText(view, product_names[class_id], (int(x1 * scale), max(int(y1 * scale) - 5, 14)), FONT,
                    config.FONT_SCALE, color, config.TEXT_THICKNESS + 1)
    lines = [f"{'>' if i == current else ' '} {i + 1}  {name}" for i, name in enumerate(product_names)]
    lines += ["", "Klik benda: kotak otomatis", "Tarik mouse: kotak sendiri", "Z batal   N lanjut",
              "X buang foto   Q selesai", "", message]
    panel = np.full((view.shape[0], config.PANEL_WIDTH, 3), config.COLOR_PANEL, dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (config.PANEL_MARGIN, config.PANEL_MARGIN + config.LINE_HEIGHT * (i + 1)),
                    FONT, config.FONT_SCALE, config.COLOR_TEXT, config.TEXT_THICKNESS)
    return np.hstack([view, panel])


def handle_mouse(state, event, x, y):
    """Klik membuat kotak lewat FastSAM. Menarik mouse membuat kotak sendiri."""
    scale = config.CLICK_SCALE
    if event == cv2.EVENT_LBUTTONDOWN:
        state["down"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["down"] is not None:
        start, state["down"] = state["down"], None
        if state["current"] is None:
            state["message"] = "Pilih produk dulu: tekan angka 1 sampai 8"
            return
        if max(abs(x - start[0]), abs(y - start[1])) >= config.CLICK_DRAG_MIN:
            x1, x2 = sorted((int(start[0] / scale), int(x / scale)))
            y1, y2 = sorted((int(start[1] / scale), int(y / scale)))
            box = (x1, y1, x2, y2)
        else:
            box = mask_box(state["model"], state["image"], int(x / scale), int(y / scale))
        if box is None:
            state["message"] = "Kotak tidak ketemu. Klik tepat di tengah benda"
            return
        state["boxes"].append((state["current"], *box))
        state["message"] = ""


def main():
    product_names = [product["name"] for product in load_products(config.PRODUCTS_CSV).values()]
    photos = [photo for photo in sorted(SCENES_DIR.glob("*.jpg")) if not photo.with_suffix(".txt").exists()]
    if not photos:
        print("Tidak ada foto yang perlu dilabeli.")
        return

    state = {"model": FastSAM(str(config.SAM_MODEL_PATH)), "down": None, "current": None}
    cv2.namedWindow(config.WINDOW_TITLE)
    cv2.setMouseCallback(config.WINDOW_TITLE, lambda event, x, y, flags, param: handle_mouse(state, event, x, y))

    saved = 0
    for photo in photos:
        state.update(image=cv2.imread(str(photo)), boxes=[], message="")
        while True:
            cv2.imshow(config.WINDOW_TITLE, draw_view(state["image"], state["boxes"], product_names, state["current"], state["message"]))
            key = cv2.waitKey(30) & 0xFF
            key = chr(key).lower() if key != 255 else ""
            if key.isdigit() and 1 <= int(key) <= len(product_names):
                state["current"] = int(key) - 1
            elif key == "z" and state["boxes"]:
                state["boxes"].pop()
            elif key == "x":
                (SCENES_DIR / "dibuang").mkdir(exist_ok=True)
                photo.rename(SCENES_DIR / "dibuang" / photo.name)
                break
            elif key in ("n", " ", "q"):
                if state["boxes"]:
                    write_labels(photo, state["boxes"], state["image"].shape)
                    saved += 1
                break
        if key == "q":
            break
    cv2.destroyAllWindows()
    print(f"Selesai. {saved} foto disimpan.")


if __name__ == "__main__":
    main()
