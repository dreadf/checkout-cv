import config


def detect(model, frame):
    """Jalankan YOLO. Hasilnya list dict: box (x1, y1, x2, y2), class_name, conf."""
    # Frame tetap BGR. Ultralytics mengubah ke RGB sendiri di dalamnya.
    result = model(frame, imgsz=config.IMG_SIZE, conf=config.CONF_FLOOR, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        detections.append({
            "box": (x1, y1, x2, y2),
            "class_name": result.names[int(box.cls[0])],
            "conf": float(box.conf[0]),
        })
    return detections
