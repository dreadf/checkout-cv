from collections import Counter

import cv2
import numpy as np

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def format_rupiah(amount):
    """12000 menjadi 'Rp 12.000'."""
    return "Rp " + f"{amount:,}".replace(",", ".")


def is_recognised(det, products):
    """Produk dikenali kalau namanya ada di products.csv dan confidence-nya cukup tinggi."""
    return det["class_name"] in products and det["conf"] >= config.CONF_OK


def class_color(class_name, products):
    return config.CLASS_COLORS[list(products).index(class_name) % len(config.CLASS_COLORS)]


def draw_detections(view, detections, products):
    """Gambar kotak dan label di frame yang sudah diperbesar. Tiap produk punya warna sendiri, merah kalau tidak dikenal."""
    for det in detections:
        x1, y1, x2, y2 = (int(v * config.DISPLAY_SCALE) for v in det["box"])
        if is_recognised(det, products):
            color = class_color(det["class_name"], products)
            label = f"{products[det['class_name']]['name']} {det['conf']:.2f}"
            cv2.rectangle(view, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, config.FONT_SCALE, config.TEXT_THICKNESS)
            top = max(y1 - text_h - baseline, 0)
            cv2.rectangle(view, (x1, top), (x1 + text_w, top + text_h + baseline), color, -1)
            cv2.putText(view, label, (x1, top + text_h), FONT, config.FONT_SCALE,
                        config.COLOR_LABEL_TEXT, config.TEXT_THICKNESS)
        else:
            cv2.rectangle(view, (x1, y1), (x2, y2), config.COLOR_PROBLEM, 1)
            cv2.putText(view, f"Tidak dikenal {det['conf']:.2f}", (x1, max(y1 - 4, 12)), FONT,
                        config.FONT_SCALE, config.COLOR_PROBLEM, config.TEXT_THICKNESS)
    return view


def summarize(detections, products):
    """Hitung barang per produk, total harga, dan jumlah yang tidak dikenal."""
    known = [d["class_name"] for d in detections if is_recognised(d, products)]
    counts = Counter(known)
    total = sum(products[name]["price"] * qty for name, qty in counts.items())
    unknown = len(detections) - len(known)
    return counts, total, unknown


def build_panel(height, counts, total, unknown, products, fps):
    """Buat panel di sebelah kanan: FPS, daftar belanja, total."""
    lines = []

    def add(text, color=config.COLOR_TEXT, scale=config.FONT_SCALE):
        lines.append((text, color, scale))

    add("Self-Checkout", scale=config.FONT_SCALE_LARGE)
    add(f"FPS: {fps:.1f}")
    add("Kunci transaksi belum aktif.", config.COLOR_WARN)
    add("")
    add("Daftar belanja", scale=config.FONT_SCALE_LARGE)
    for class_name, qty in counts.items():
        item = products[class_name]
        add(f"{qty} x {item['name']}")
        add(f"      {format_rupiah(item['price'] * qty)}")
    if unknown:
        add(f"Tidak dikenal: {unknown}", config.COLOR_PROBLEM)
    add("")
    add(f"TOTAL {format_rupiah(total)}", scale=config.FONT_SCALE_LARGE)

    # Panel dibuat lebih tinggi dari frame kalau isinya tidak muat.
    needed = 2 * config.PANEL_MARGIN + (len(lines) + 1) * config.LINE_HEIGHT
    height = max(height, needed)
    panel = np.full((height, config.PANEL_WIDTH, 3), config.COLOR_PANEL, dtype=np.uint8)
    y = config.PANEL_MARGIN + config.LINE_HEIGHT
    for text, color, scale in lines:
        cv2.putText(panel, text, (config.PANEL_MARGIN, y), FONT, scale, color, config.TEXT_THICKNESS)
        y += config.LINE_HEIGHT
    cv2.putText(panel, "Tekan Q untuk keluar", (config.PANEL_MARGIN, height - config.PANEL_MARGIN),
                FONT, config.FONT_SCALE, config.COLOR_TEXT, config.TEXT_THICKNESS)
    return panel


def compose(frame, panel):
    """Tempel panel di kanan gambar kamera. Frame diberi bingkai bawah kalau panel lebih tinggi."""
    extra = panel.shape[0] - frame.shape[0]
    if extra > 0:
        frame = cv2.copyMakeBorder(frame, 0, extra, 0, 0, cv2.BORDER_CONSTANT, value=config.COLOR_PANEL)
    return np.hstack([frame, panel])
