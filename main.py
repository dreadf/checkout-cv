import time

import cv2
from ultralytics import YOLO

import config
from app import camera, checks, detector, display, lock, logger, stabilize
from app.products import load_products

QUIT_KEYS = (ord("q"), 27)
ENTER_KEYS = (13, 10)
VIEW_WIDTH = round(config.PROCESS_WIDTH * config.DISPLAY_SCALE)


def handle_click(mouse_state, event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN and mouse_state["button"] is not None:
        # Tombol digambar di panel, yang ditempel di sebelah kanan gambar kamera.
        point = (x - VIEW_WIDTH, y)
        if display.button_hit(point, mouse_state["button"]):
            mouse_state["clicked"] = True


def main():
    products = load_products(config.PRODUCTS_CSV)
    model = YOLO(str(config.MODEL_PATH))

    cap = camera.open_camera()
    if cap is None:
        print("Kamera tidak bisa dibuka. Cek CAMERA_INDEX di config.py dan izin kamera di sistem.")
        return

    mouse_state = {"button": None, "clicked": False}
    cv2.namedWindow(config.WINDOW_TITLE)
    cv2.setMouseCallback(config.WINDOW_TITLE, lambda e, x, y, f, p: handle_click(mouse_state, e, x, y))

    tracked_items = {}
    streaks = {}
    background = None
    previous_frame = None
    state = "KOSONG"
    calm_since = None
    last_signature = None
    lock_events = 0
    seconds_locked = 0.0

    fps = 0.0
    last_time = time.perf_counter()
    while True:
        ok, raw = cap.read()
        if not ok:
            print("Gagal membaca gambar dari kamera.")
            break
        now = time.perf_counter()
        dt = now - last_time
        fps = config.FPS_SMOOTHING * fps + (1 - config.FPS_SMOOTHING) / dt if dt > 0 else fps
        last_time = now

        frame = camera.preprocess(raw)
        detections = checks.remove_overlaps(detector.detect(model, frame))
        stable_items = checks.remove_overlaps(stabilize.update_items(tracked_items, detections))
        # Kotak merah hanya untuk deteksi ragu-ragu yang tidak menimpa barang yang sudah dikenali.
        unrecognised = [d for d in detections if d["conf"] < config.CONF_OK
                        and all(checks.iou(d["box"], item["box"]) <= config.OVERLAP_IOU for item in stable_items)]
        counts, total, unknown = display.summarize(stable_items, len(unrecognised), products)
        known_boxes = [item["box"] for item in stable_items]

        cam_problem = camera.camera_problem(frame, previous_frame)
        use_blobs = config.CHECK_OBJECT_BLOBS and background is not None and cam_problem is None
        blobs = checks.find_object_blobs(frame, background) if use_blobs else []
        conditions = {"kamera": cam_problem, "belum_kalibrasi": config.CHECK_OBJECT_BLOBS and background is None}
        if cam_problem is None and (use_blobs or not config.CHECK_OBJECT_BLOBS):
            conditions.update(checks.table_conditions(frame, previous_frame, blobs, known_boxes))

        # Selesai ditolak selama ada benda asing atau gerak yang belum lolos penyaring kedipan, atau selama total
        # masih memuat barang yang baru hilang, supaya struk tidak tercetak untuk keadaan meja yang sudah berubah.
        unsettled = any(conditions.get(key) for key in ("tidak_dikenal", "tepi", "bergerak"))
        conditions = lock.confirmed(conditions, streaks, config.LOCK_CONFIRM_FRAMES)
        has_items = bool(detections) or bool(conditions["tidak_dikenal"]) or bool(conditions["tepi"])
        problem = any(conditions.values())

        # Barang yang baru muncul tapi belum stabil ikut dihitung perubahan lewat jumlah barang yang dilacak.
        signature = (tuple(sorted(counts.items())), len(tracked_items))
        if problem or not has_items or signature != last_signature:
            calm_since = now if (has_items and not problem) else None
        elif calm_since is None:
            calm_since = now
        last_signature = signature
        stable_seconds = (now - calm_since) if calm_since else 0.0

        if state != "TERKUNCI" and problem:
            lock_events += 1
        if state == "TERKUNCI":
            seconds_locked += dt

        new_state = lock.next_state(state, has_items, problem, stable_seconds, config.STILL_SECONDS)
        if new_state == "KOSONG" and state != "KOSONG":
            lock_events, seconds_locked = 0, 0.0
        state = new_state
        reasons = lock.message_for(conditions)

        view = cv2.resize(frame, None, fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE)
        display.draw_detections(view, stable_items, unrecognised, products)
        display.draw_unexplained(view, checks.uncovered_blobs(blobs, known_boxes))
        panel, button = display.build_panel(view.shape[0], counts, total, unknown, products, fps, state,
                                            reasons, lock.hint_for(state))
        mouse_state["button"] = button
        cv2.imshow(config.WINDOW_TITLE, display.compose(view, panel))

        key = cv2.waitKey(1) & 0xFF
        finish = mouse_state["clicked"] or key in ENTER_KEYS
        mouse_state["clicked"] = False
        settled = not unsettled and all(item["lost"] == 0 for item in stable_items)
        if finish and state == "SIAP" and settled:
            logger.print_receipt(counts, total, products)
            logger.log_transaction(counts, total, lock_events, seconds_locked)
            state = "SELESAI"
        elif key == ord("k"):
            captured = []
            for _ in range(config.BACKGROUND_CAPTURE_FRAMES):
                ok2, raw2 = cap.read()
                if ok2:
                    captured.append(camera.preprocess(raw2))
            if captured:
                background = checks.empty_table_reference(captured)
                print("Referensi meja kosong disimpan.")
        elif key in QUIT_KEYS:
            break

        previous_frame = frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
