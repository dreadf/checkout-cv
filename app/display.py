from collections import Counter

import cv2
import numpy as np

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def format_rupiah(amount):
    """12000 menjadi 'Rp 12.000'."""
    return "Rp " + f"{amount:,}".replace(",", ".")


def class_color(class_name, products):
    return config.CLASS_COLORS[list(products).index(class_name) % len(config.CLASS_COLORS)]


def draw_detections(view, known_items, unknown_detections, products):
    """Gambar kotak dan label di frame yang sudah diperbesar. known_items sudah stabil (app/stabilize.py),
    unknown_detections digambar merah apa adanya setiap frame (confidence rendah dari YOLO, belum stabil)."""
    for item in known_items:
        x1, y1, x2, y2 = (int(v * config.DISPLAY_SCALE) for v in item["box"])
        color = class_color(item["class_name"], products)
        label = f"{products[item['class_name']]['name']} {item['conf']:.2f}"
        cv2.rectangle(view, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, config.FONT_SCALE, config.TEXT_THICKNESS)
        top = max(y1 - text_h - baseline, 0)
        cv2.rectangle(view, (x1, top), (x1 + text_w, top + text_h + baseline), color, -1)
        cv2.putText(view, label, (x1, top + text_h), FONT, config.FONT_SCALE,
                    config.COLOR_LABEL_TEXT, config.TEXT_THICKNESS)
    for det in unknown_detections:
        x1, y1, x2, y2 = (int(v * config.DISPLAY_SCALE) for v in det["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), config.COLOR_PROBLEM, 1)
        cv2.putText(view, f"Tidak dikenal {det['conf']:.2f}", (x1, max(y1 - 4, 12)), FONT,
                    config.FONT_SCALE, config.COLOR_PROBLEM, config.TEXT_THICKNESS)
    return view


def draw_unexplained(view, blobs):
    """Gambar kotak putih di benda yang terlihat berbeda dari meja kosong tapi tidak dijelaskan kotak produk
    mana pun. Inilah yang dihitung sebagai "benda tidak dikenali" pada alasan kunci."""
    for blob in blobs:
        x1, y1, x2, y2 = (int(v * config.DISPLAY_SCALE) for v in blob["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), config.COLOR_BLOB, 1)
        cv2.putText(view, "benda?", (x1 + 3, y1 + 16), FONT, config.FONT_SCALE, config.COLOR_BLOB, config.TEXT_THICKNESS)
    return view


def summarize(known_items, unknown_count, products):
    """Hitung barang per produk dan total harga dari daftar barang yang sudah stabil.
    unknown_count dihitung terpisah oleh pemanggil, dari deteksi confidence rendah."""
    counts = Counter(item["class_name"] for item in known_items)
    total = sum(products[name]["price"] * qty for name, qty in counts.items())
    return counts, total, unknown_count


def draw_padlock(panel, x, y):
    """Gambar ikon gembok kecil di (x, y), badan kotak dan lengkung di atasnya."""
    cv2.rectangle(panel, (x, y), (x + 18, y + 14), config.COLOR_TEXT, -1)
    cv2.ellipse(panel, (x + 9, y - 1), (7, 9), 0, 180, 360, config.COLOR_TEXT, 2)


def button_rect(height):
    """Posisi dan ukuran tombol Selesai di panel, dari bawah."""
    x = config.PANEL_MARGIN
    y = height - config.PANEL_MARGIN - config.LINE_HEIGHT - config.BUTTON_HEIGHT
    return x, y, config.BUTTON_WIDTH, config.BUTTON_HEIGHT


def draw_finish_button(panel, height, locked):
    """Tombol Selesai: hijau aktif, abu-abu dengan gembok kalau terkunci. Alasannya ada di blok di atasnya."""
    x, y, w, h = button_rect(height)
    color = config.COLOR_PANEL_LOCKED if locked else config.COLOR_PANEL_READY
    cv2.rectangle(panel, (x, y), (x + w, y + h), color, -1)
    label = "TERKUNCI" if locked else "SELESAI"
    (text_w, text_h), _ = cv2.getTextSize(label, FONT, config.FONT_SCALE_LARGE, config.TEXT_THICKNESS + 1)
    offset = 16 if locked else 0
    text_x = x + (w - text_w) // 2 - offset
    text_y = y + h // 2 + text_h // 2
    cv2.putText(panel, label, (text_x, text_y), FONT, config.FONT_SCALE_LARGE,
                config.COLOR_TEXT, config.TEXT_THICKNESS + 1)
    if locked:
        draw_padlock(panel, text_x + text_w + 10, y + h // 2 - 6)
    return (x, y, w, h)


def wrap_text(text, max_width):
    """Pecah teks jadi beberapa baris supaya tiap baris muat selebar max_width piksel."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if cv2.getTextSize(candidate, FONT, config.FONT_SCALE, config.TEXT_THICKNESS)[0][0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    return lines + [current] if current else lines


def button_hit(point, rect):
    """True kalau titik klik (x, y) jatuh di dalam rect tombol."""
    px, py = point
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def build_panel(height, counts, total, unknown, products, fps, state, reasons, hint):
    """Buat panel di sebelah kanan: FPS, status, daftar belanja, total, alasan kunci, dan tombol Selesai.
    reasons: list (masalah, tindakan) dari app/lock.py. hint: petunjuk saat tidak ada masalah.
    Hasilnya (panel, button_rect)."""
    lines = []
    text_width = config.PANEL_WIDTH - 2 * config.PANEL_MARGIN

    def add(text, color=config.COLOR_TEXT, scale=config.FONT_SCALE):
        lines.append((text, color, scale))

    def add_wrapped(text, color, indent=""):
        for line in wrap_text(text, text_width - len(indent) * 8):
            add(indent + line, color)

    add("Self-Checkout", scale=config.FONT_SCALE_LARGE)
    add(f"FPS: {fps:.1f}")
    add(f"Status: {state}", config.COLOR_WARN)
    add("")
    add("Daftar belanja", scale=config.FONT_SCALE_LARGE)
    for class_name, qty in counts.items():
        item = products[class_name]
        add(f"{qty} x {item['name']}")
        add(f"      {format_rupiah(item['price'] * qty)}")
    if unknown:
        add(f"Kotak ragu-ragu (merah): {unknown}", config.COLOR_REASON)
    add("")
    add(f"TOTAL {format_rupiah(total)}", scale=config.FONT_SCALE_LARGE)
    add("")
    if reasons:
        add("Kenapa terkunci", scale=config.FONT_SCALE_LARGE)
        for problem, action in reasons:
            add_wrapped(problem, config.COLOR_REASON)
            add_wrapped("Lakukan: " + action, config.COLOR_TEXT, indent="  ")
    elif hint:
        add_wrapped(hint, config.COLOR_PANEL_READY if state == "SIAP" else config.COLOR_TEXT)

    # Ruang dicadangkan di bawah untuk tombol Selesai dan baris bantuan tombol.
    reserved = config.BUTTON_HEIGHT + 2 * config.LINE_HEIGHT
    needed = 2 * config.PANEL_MARGIN + (len(lines) + 1) * config.LINE_HEIGHT + reserved
    height = max(height, needed)
    panel = np.full((height, config.PANEL_WIDTH, 3), config.COLOR_PANEL, dtype=np.uint8)
    y = config.PANEL_MARGIN + config.LINE_HEIGHT
    for text, color, scale in lines:
        cv2.putText(panel, text, (config.PANEL_MARGIN, y), FONT, scale, color, config.TEXT_THICKNESS)
        y += config.LINE_HEIGHT

    button = draw_finish_button(panel, height, state != "SIAP")
    cv2.putText(panel, "K: referensi meja kosong   Q: keluar", (config.PANEL_MARGIN, height - 8),
                FONT, config.FONT_SCALE, config.COLOR_TEXT, config.TEXT_THICKNESS)
    return panel, button


def compose(frame, panel):
    """Tempel panel di kanan gambar kamera. Frame diberi bingkai bawah kalau panel lebih tinggi."""
    extra = panel.shape[0] - frame.shape[0]
    if extra > 0:
        frame = cv2.copyMakeBorder(frame, 0, extra, 0, 0, cv2.BORDER_CONSTANT, value=config.COLOR_PANEL)
    return np.hstack([frame, panel])
