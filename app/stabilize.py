"""Menstabilkan deteksi antar frame supaya daftar belanja tidak berkedip.

Barang baru dihitung setelah terlihat STABLE_FRAMES frame berturut-turut, dan dibuang setelah
hilang LOST_FRAMES frame berturut-turut. Tidak memakai library tracker, hanya mencocokkan kelas
yang sama dan titik tengah terdekat.
"""

import config


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _nearest(item, detections, max_dist):
    """Deteksi sekelas yang paling dekat dengan barang, di bawah max_dist. None kalau tidak ada."""
    best, best_dist = None, max_dist
    for det in detections:
        if det["class_name"] != item["class_name"]:
            continue
        dist = _distance(_center(det["box"]), _center(item["box"]))
        if dist < best_dist:
            best, best_dist = det, dist
    return best


def update_items(items, detections):
    """Cocokkan deteksi frame ini dengan barang yang sudah dilacak.

    items: dict id -> {"class_name", "box", "conf", "seen", "lost"}, diubah di tempat.
    Hasilnya daftar barang yang sudah stabil (seen >= STABLE_FRAMES), termasuk yang baru hilang
    sebentar (lost > 0). Pemanggil menggambar hanya yang lost == 0.

    Pencocokan dua putaran. Putaran pertama: deteksi sekelas yang dekat (di bawah MATCH_MAX_DIST).
    Putaran kedua: barang yang masih hilang mengambil deteksi baru sekelas di mana pun, jadi botol
    yang dipindah dianggap barang yang sama, bukan barang baru yang dihitung dobel.

    detections berisi semua deteksi di atas CONF_FLOOR. Barang yang sudah dilacak terus cocok dengan
    deteksi sekelas yang dekat selama confidence-nya minimal CONF_KEEP, jadi tidak berganti antara hijau dan
    merah saat confidence turun sebentar di bawah CONF_OK. Tebakan yang lebih lemah dari itu tidak menahan
    barang yang sudah diangkat. Barang baru hanya dibuat dari deteksi dengan confidence minimal CONF_OK.
    """
    unmatched = list(detections)
    for item in items.values():
        item["lost"] += 1

    # Putaran kedua (di mana pun) hanya boleh mengambil deteksi yang yakin, supaya tebakan lemah di tempat lain
    # tidak menghidupkan lagi barang yang sudah hilang.
    for max_dist, min_conf in ((config.MATCH_MAX_DIST, config.CONF_KEEP), (float("inf"), config.CONF_OK)):
        pool = [d for d in unmatched if d["conf"] >= min_conf]
        for item in items.values():
            if item["lost"] == 0:
                continue
            best = _nearest(item, pool, max_dist)
            if best is not None:
                item.update(box=best["box"], conf=best["conf"], seen=item["seen"] + 1, lost=0)
                unmatched.remove(best)
                pool.remove(best)

    for det in unmatched:
        if det["conf"] < config.CONF_OK:
            continue
        new_id = max(items, default=0) + 1
        items[new_id] = {"class_name": det["class_name"], "box": det["box"], "conf": det["conf"], "seen": 1, "lost": 0}

    for item_id in [i for i, item in items.items() if item["lost"] > config.LOST_FRAMES]:
        del items[item_id]

    return [item for item in items.values() if item["seen"] >= config.STABLE_FRAMES]
