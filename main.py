import time

import cv2
from ultralytics import YOLO

import config
from app import camera, detector, display
from app.products import load_products

QUIT_KEYS = (ord("q"), 27)


def main():
    products = load_products(config.PRODUCTS_CSV)
    model = YOLO(str(config.MODEL_PATH))

    cap = camera.open_camera()
    if cap is None:
        print("Kamera tidak bisa dibuka. Cek CAMERA_INDEX di config.py dan izin kamera di sistem.")
        return

    fps = 0.0
    last_time = time.perf_counter()
    while True:
        ok, raw = cap.read()
        if not ok:
            print("Gagal membaca gambar dari kamera.")
            break

        frame = camera.preprocess(raw)
        detections = detector.detect(model, frame)
        counts, total, unknown = display.summarize(detections, products)
        view = cv2.resize(frame, None, fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE)
        display.draw_detections(view, detections, products)

        now = time.perf_counter()
        fps = config.FPS_SMOOTHING * fps + (1 - config.FPS_SMOOTHING) / (now - last_time)
        last_time = now

        panel = display.build_panel(view.shape[0], counts, total, unknown, products, fps)
        cv2.imshow(config.WINDOW_TITLE, display.compose(view, panel))
        if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
