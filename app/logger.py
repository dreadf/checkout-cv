"""Mencatat transaksi ke CSV dan menampilkan struk sederhana di terminal. Dipanggil saat Selesai."""

import csv
from datetime import datetime

import config


def log_transaction(counts, total, lock_events, seconds_locked):
    """Tambah satu baris ke logs/transactions.csv. File dan judul kolom dibuat kalau belum ada."""
    path = config.LOGS_DIR / "transactions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["waktu", "barang", "total", "kejadian_kunci", "detik_terkunci"])
        barang = "|".join(f"{name}:{qty}" for name, qty in counts.items())
        writer.writerow([datetime.now().isoformat(timespec="seconds"), barang, total,
                          lock_events, round(seconds_locked, 1)])


def print_receipt(counts, total, products):
    """Cetak struk sederhana ke terminal. Hanya print, tidak ada file atau tampilan tambahan."""
    print("===== STRUK =====")
    print(datetime.now().strftime("%d %b %Y  %H:%M"))
    for class_name, qty in counts.items():
        item = products[class_name]
        subtotal = f"Rp {item['price'] * qty:,}".replace(",", ".")
        print(f"{qty} x {item['name']:<24} {subtotal}")
    print("-------------------------------")
    total_text = f"Rp {total:,}".replace(",", ".")
    print(f"TOTAL{'':<21} {total_text}")
    print("===============================")
