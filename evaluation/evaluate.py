"""Ukur seberapa baik model mengenali barang pada foto tes (samples/).

Cara pakai dari folder utama repo:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --model best_v1.pt --conf 0.5

Foto tes dibandingkan dengan daftar barang yang benar di scenes.csv. Perbandingannya berdasarkan
jumlah barang per produk, bukan posisi kotak, karena foto tes hanya punya daftar barang.
Hasilnya ditulis ke evaluation/results/<tanggal>_<nama model>/.
"""

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from app import checks
from app.detector import detect
from app.products import load_products

TEST_DIR = config.SAMPLES_DIR
MISSED = "(tidak terdeteksi)"
EXTRA = "(benda ekstra)"


def ratio(top, bottom):
    """Pembagian yang aman. Hasilnya None kalau pembaginya nol."""
    return top / bottom if bottom else None


def fmt(value):
    return "" if value is None else round(value, 3)


def load_scenes():
    """Baca scenes.csv. Hasilnya list dict: file, truth (jumlah benar per produk), category."""
    scenes = []
    with open(TEST_DIR / "scenes.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = Counter()
            for entry in row["items"].split("|"):
                name, _, qty = entry.partition(":")
                truth[name] += int(qty) if qty else 1
            scenes.append({"file": row["file"], "truth": truth, "category": row["category"],
                           "should_lock": row["should_lock"].strip().lower() == "yes"})
    return scenes


def count_items(detections, conf):
    """Jumlah barang per produk dari kotak yang confidence-nya minimal conf."""
    return Counter(d["class_name"] for d in detections if d["conf"] >= conf)


def compare(truth, predicted):
    """Bandingkan jumlah benar dan jumlah tebakan satu adegan. Hasilnya (tp, fp, fn) per produk."""
    names = sorted(set(truth) | set(predicted))
    tp = {name: min(truth[name], predicted[name]) for name in names}
    fp = {name: max(predicted[name] - truth[name], 0) for name in names}
    fn = {name: max(truth[name] - predicted[name], 0) for name in names}
    return tp, fp, fn


def pair_swaps(fp, fn):
    """Pasangkan barang yang kurang dengan barang yang berlebih di adegan yang sama, anggap tertukar.

    Hasilnya (swaps, missed, extra). swaps: (benar, tebakan) -> jumlah. missed dan extra: sisa yang tidak
    bisa dipasangkan, yaitu barang yang tidak terdeteksi dan benda ekstra.
    """
    missing, extra = dict(fn), dict(fp)
    swaps = Counter()
    for true_name in missing:
        for guessed in extra:
            n = min(missing[true_name], extra[guessed])
            if n:
                swaps[(true_name, guessed)] += n
                missing[true_name] -= n
                extra[guessed] -= n
    return swaps, missing, extra


def score(scenes, predictions, conf, products):
    """Hitung semua ukuran untuk satu ambang. Hasilnya (baris per adegan, total per produk, confusion)."""
    rows, per_product, confusion = [], {name: Counter() for name in products}, Counter()
    for scene, detections in zip(scenes, predictions):
        predicted = count_items(detections, conf)
        tp, fp, fn = compare(scene["truth"], predicted)
        swaps, missed, extra = pair_swaps(fp, fn)
        for name in tp:
            per_product[name].update(tp=tp[name], fp=fp[name], fn=fn[name])
            confusion[(name, name)] += tp[name]
            confusion[(name, MISSED)] += missed[name]
            confusion[(EXTRA, name)] += extra[name]
        confusion.update(swaps)
        truth_total = sum(products[name]["price"] * qty for name, qty in scene["truth"].items())
        predicted_total = sum(products[name]["price"] * qty for name, qty in predicted.items())
        rows.append({
            "file": scene["file"], "category": scene["category"],
            "truth": scene["truth"], "predicted": predicted,
            "tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values()),
            "exact_list": scene["truth"] == predicted,
            "truth_total": truth_total, "predicted_total": predicted_total,
            "total_error": abs(predicted_total - truth_total),
        })
    return rows, per_product, confusion


def summarize(rows):
    """Ringkasan sekumpulan adegan: precision, recall, persen daftar persis, persen total tepat, selisih Rp."""
    n = len(rows)
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    return {
        "adegan": n,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "daftar_persis": ratio(sum(row["exact_list"] for row in rows), n),
        "total_tepat": ratio(sum(row["total_error"] == 0 for row in rows), n),
        "selisih_rp": ratio(sum(row["total_error"] for row in rows), n),
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def score_lock(scenes, predictions, background, conf):
    """Bandingkan prediksi kunci (benda asing atau barang di tepi) dengan should_lock di scenes.csv.

    Cek gerakan dan kesehatan kamera butuh dua frame berurutan, jadi tidak bisa diuji jujur dari
    foto diam. Hanya benda asing dan tepi yang dihitung di sini. Hasilnya (rows, ringkasan).
    """
    rows = []
    for scene, detections in zip(scenes, predictions):
        frame = cv2.imread(str(TEST_DIR / scene["file"]))
        known_boxes = [d["box"] for d in detections if d["conf"] >= conf]
        blobs = checks.find_object_blobs(frame, background)
        conditions = checks.table_conditions(frame, None, blobs, known_boxes)
        locked = bool(conditions["tidak_dikenal"] or conditions["tepi"])
        rows.append({"file": scene["file"], "category": scene["category"],
                      "should_lock": scene["should_lock"], "locked": locked})
    tp = sum(r["locked"] and r["should_lock"] for r in rows)
    fn = sum((not r["locked"]) and r["should_lock"] for r in rows)
    fp = sum(r["locked"] and (not r["should_lock"]) for r in rows)
    tn = sum((not r["locked"]) and (not r["should_lock"]) for r in rows)
    summary = {"benar_terkunci": ratio(tp, tp + fn), "salah_kunci": ratio(fp, fp + tn)}
    return rows, summary


def plot_confusion(path, confusion, names):
    """Gambar confusion matrix. Baris: yang sebenarnya. Kolom: tebakan."""
    rows, cols = names + [EXTRA], names + [MISSED]
    matrix = np.array([[confusion[(r, c)] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Tebakan")
    ax.set_ylabel("Yang sebenarnya")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if matrix[i, j]:
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model pada foto tes.")
    parser.add_argument("--model", default=str(config.MODEL_PATH), help="Lokasi file model")
    parser.add_argument("--conf", type=float, default=config.CONF_OK, help="Ambang confidence utama")
    args = parser.parse_args()

    products = load_products(config.PRODUCTS_CSV)
    names = list(products)
    scenes = load_scenes()
    model = YOLO(args.model)
    # Sama seperti aplikasi: kotak kembar dibuang dulu sebelum dihitung.
    predictions = [checks.remove_overlaps(detect(model, cv2.imread(str(TEST_DIR / scene["file"]))))
                   for scene in scenes]

    out_dir = config.EVAL_RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_{Path(args.model).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, per_product, confusion = score(scenes, predictions, args.conf, products)
    write_csv(out_dir / "per_produk.csv", ["produk", "benar", "salah_ada", "terlewat", "precision", "recall"],
              [[name, c["tp"], c["fp"], c["fn"], fmt(ratio(c["tp"], c["tp"] + c["fp"])),
                fmt(ratio(c["tp"], c["tp"] + c["fn"]))] for name, c in per_product.items()])
    write_csv(out_dir / "per_adegan.csv",
              ["foto", "kategori", "daftar_benar", "daftar_tebakan", "daftar_persis", "total_benar", "total_tebakan", "selisih_rp"],
              [[r["file"], r["category"], "|".join(f"{k}:{v}" for k, v in r["truth"].items()),
                "|".join(f"{k}:{v}" for k, v in r["predicted"].items()), r["exact_list"],
                r["truth_total"], r["predicted_total"], r["total_error"]] for r in rows])
    categories = sorted({r["category"] for r in rows})
    groups = [("semua", rows)] + [(c, [r for r in rows if r["category"] == c]) for c in categories]
    header = ["kategori", "adegan", "precision", "recall", "daftar_persis", "total_tepat", "selisih_rp_rata"]
    write_csv(out_dir / "per_kategori.csv", header,
              [[name] + [fmt(v) for v in summarize(group).values()] for name, group in groups])
    sweep = []
    for threshold in config.EVAL_THRESHOLDS:
        sweep_rows, _, _ = score(scenes, predictions, threshold, products)
        sweep.append([threshold] + [fmt(v) for v in list(summarize(sweep_rows).values())[1:]])
    write_csv(out_dir / "ambang.csv", ["ambang", "precision", "recall", "daftar_persis", "total_tepat", "selisih_rp_rata"], sweep)
    write_csv(out_dir / "confusion_matrix.csv", ["sebenarnya"] + names + [MISSED],
              [[r] + [confusion[(r, c)] for c in names + [MISSED]] for r in names + [EXTRA]])
    plot_confusion(out_dir / "confusion_matrix.png", confusion, names)

    photos_dir = config.DATA_DIR / "product_photos"
    if photos_dir.exists():
        from tools.auto_label import build_background
        photos = [p for folder in sorted(photos_dir.iterdir()) if folder.is_dir() for p in sorted(folder.glob("*.jpg"))]
        background = build_background(photos)
        kunci_rows, kunci_summary = score_lock(scenes, predictions, background, args.conf)
        write_csv(out_dir / "kunci.csv", ["foto", "kategori", "should_lock", "terkunci"],
                  [[r["file"], r["category"], r["should_lock"], r["locked"]] for r in kunci_rows])
        print(f"Kunci: benar terkunci {fmt(kunci_summary['benar_terkunci'])}, "
              f"salah kunci {fmt(kunci_summary['salah_kunci'])} (hanya benda asing dan tepi, "
              f"gerakan dan kamera tidak bisa diuji dari foto diam)")
    else:
        print("data/product_photos tidak ada, lewati pengukuran kunci (kunci.csv tidak dibuat)")

    total = summarize(rows)
    print(f"Model: {args.model}")
    print(f"Ambang confidence: {args.conf}, {total['adegan']} adegan")
    print(f"Precision {fmt(total['precision'])}, recall {fmt(total['recall'])}")
    print(f"Daftar persis benar {fmt(total['daftar_persis'])}, total harga tepat {fmt(total['total_tepat'])}, "
          f"selisih rata-rata Rp {round(total['selisih_rp'])}")
    print(f"Hasil lengkap di {out_dir}")


if __name__ == "__main__":
    main()
