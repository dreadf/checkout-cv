"""State machine kunci transaksi. Fungsi murni, tidak membaca kamera, bisa dites dengan data palsu.

Status: KOSONG, MEMERIKSA, TERKUNCI, SIAP, SELESAI. Status KALIBRASI di desain awal digabung ke
KOSONG lewat kondisi "belum_kalibrasi" (tombol K belum pernah ditekan), supaya jumlah status tetap
sedikit dan mudah dijelaskan.
"""

# Setiap kondisi punya dua kalimat: apa masalahnya, dan apa yang harus dilakukan pembeli.
# {n} diisi jumlah benda untuk kondisi yang menghitung benda.
PESAN = {
    "belum_kalibrasi": ("Meja belum dikalibrasi", "Kosongkan meja, lalu tekan K"),
    "tidak_dikenal": ("Ada {n} benda yang tidak dikenali", "Angkat benda selain produk yang dijual"),
    "tepi": ("Ada {n} benda tidak dikenali di pinggir meja", "Geser ke tengah atau angkat dari meja"),
    "bergerak": ("Meja sedang bergerak ({n:.1%} gambar berubah)", "Jangan sentuh barang, tunggu sebentar"),
}
# Kondisi kamera membawa jenis masalahnya ("gelap", "terhalang", atau "membeku") sebagai nilainya.
PESAN_KAMERA = {
    "gelap": ("Kamera terlalu gelap", "Nyalakan lampu atau tambah cahaya"),
    "terhalang": ("Kamera tertutup", "Buka penutup lensa atau singkirkan benda di depannya"),
    "membeku": ("Gambar kamera macet", "Periksa sambungan kamera, lalu jalankan ulang aplikasi"),
}
# Kelompok kondisi yang bisa berkedip karena bayangan, pantulan, atau tangan lewat. Baru dianggap masalah
# setelah salah satunya aktif LOCK_CONFIRM_FRAMES frame berturut-turut. "tidak_dikenal" dan "tepi" satu kelompok
# karena satu benda bisa berpindah di antara keduanya saat kotaknya bergeser sedikit di dekat batas
# EDGE_MARGIN. Kondisi kamera dan kalibrasi langsung dianggap.
KELOMPOK_BERKEDIP = (("tidak_dikenal", "tepi"), ("bergerak",))
# Urutan prioritas kalau beberapa kondisi terjadi bersamaan. Semuanya tetap ditampilkan, dari atas.
PRIORITAS = ("kamera", "belum_kalibrasi", "tidak_dikenal", "tepi", "bergerak")
# Petunjuk untuk pembeli saat tidak ada masalah.
PETUNJUK = {
    "KOSONG": "Taruh barang di meja",
    "MEMERIKSA": "Memeriksa barang, mohon tunggu sebentar",
    "SIAP": "Semua barang dikenali. Tekan Selesai",
    "SELESAI": "Transaksi dicatat. Angkat semua barang dari meja",
}


def next_state(state, has_items, problem, stable_seconds, still_seconds):
    """Tentukan status berikutnya.

    has_items: ada barang atau benda di meja (dari YOLO atau cek benda). problem: True kalau salah
    satu kondisi kunci aktif. stable_seconds: sudah berapa lama tanpa masalah dan tanpa perubahan
    barang (dihitung pemanggil). still_seconds: ambang diam yang dibutuhkan sebelum SIAP.
    """
    # Setelah transaksi dicatat, tetap SELESAI sampai meja dikosongkan, apa pun yang terjadi di meja.
    # Kalau tidak, masalah sesaat membuka tombol Selesai lagi dan transaksi yang sama tercatat dua kali.
    if state == "SELESAI":
        return "KOSONG" if not has_items else "SELESAI"
    if problem:
        return "TERKUNCI"
    if state == "TERKUNCI":
        return "MEMERIKSA" if has_items else "KOSONG"
    if not has_items:
        return "KOSONG"
    # SIAP dan MEMERIKSA memakai aturan yang sama: SIAP hanya bertahan selama meja tetap tenang.
    return "SIAP" if stable_seconds >= still_seconds else "MEMERIKSA"


def confirmed(conditions, streaks, need):
    """Saring kondisi yang berkedip: tiap kelompok di KELOMPOK_BERKEDIP baru dihitung setelah salah satu
    anggotanya aktif need frame berturut-turut, selain itu nilainya 0. streaks (dict kelompok -> jumlah frame
    berturut-turut) diubah di tempat."""
    result = dict(conditions)
    for group in KELOMPOK_BERKEDIP:
        active = any(conditions.get(key) for key in group)
        streaks[group] = streaks.get(group, 0) + 1 if active else 0
        for key in group:
            result[key] = conditions.get(key) if streaks[group] >= need else 0
    return result


def message_for(conditions):
    """Semua alasan kunci yang aktif, berurutan dari prioritas tertinggi. Hasilnya list (masalah, tindakan).
    Kosong kalau tidak ada masalah."""
    reasons = []
    for key in PRIORITAS:
        value = conditions.get(key)
        if not value:
            continue
        problem, action = PESAN_KAMERA[value] if key == "kamera" else PESAN[key]
        reasons.append((problem.format(n=value), action))
    return reasons


def hint_for(state):
    """Petunjuk singkat untuk pembeli saat tidak ada masalah. String kosong kalau tidak ada."""
    return PETUNJUK.get(state, "")
