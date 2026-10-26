"""Semua angka yang bisa diatur ada di file ini. Kode lain tidak boleh punya angka ajaib."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Path file
PRODUCTS_CSV = ROOT / "products.csv"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "samples"
MODEL_PATH = ROOT / "best.pt"
SAM_MODEL_PATH = ROOT / "FastSAM-s.pt"

# Kamera. Nomor 0 biasanya webcam bawaan laptop. Kalau tidak terbuka atau yang tampil bukan
# webcam yang dimaksud (misalnya kamera virtual), coba nomor lain (1, 2).
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Frame awal kamera dibuang karena biasanya masih gelap atau kabur.
CAPTURE_WARMUP_FRAMES = 15

# Area meja dalam piksel frame kamera, bentuknya (x, y, lebar, tinggi).
# None berarti seluruh frame. Dapatkan angkanya dari tools/set_table_region.py.
# Model dilatih dari foto yang dipotong ke area meja, jadi hasilnya lebih baik kalau area dipilih.
TABLE_REGION = None

# Lebar gambar setelah crop dan resize. Dipakai sama persis saat ambil foto training.
PROCESS_WIDTH = 640

# Deteksi YOLO
IMG_SIZE = 640
# Kotak dengan confidence di bawah angka ini dibuang. Dinaikkan dari 0,25 ke 0,4 supaya layar tidak
# penuh kotak merah ragu-ragu. Tidak memengaruhi barang yang dihitung, karena CONF_OK jauh di atasnya.
CONF_FLOOR = 0.4
# Dua kotak yang saling menimpa lebih dari ini (IoU, 0 sampai 1) dianggap benda yang sama.
# Yang confidence-nya lebih rendah dibuang (app/checks.py remove_overlaps).
OVERLAP_IOU = 0.5
# Barang baru dikenali (dan dihitung) kalau confidence-nya di atas atau sama dengan angka ini. Barang yang sudah dikenali
# bertahan sampai CONF_KEEP.
# Dinaikkan dari 0,5 ke 0,65 lewat evaluate.py --conf (precision 0,929 ke 0,957, recall 0,886 ke 0,848).
# Sengaja memilih precision lebih tinggi karena barang yang tidak lolos tetap aman: jadi "tidak dikenal",
# tidak dihitung, dan bisa memicu kunci, bukan hilang begitu saja. Di atas 0,7 recall mulai jatuh tajam.
CONF_OK = 0.65
# Barang yang sudah dilacak tetap dihitung selama ada deteksi sekelas di dekatnya dengan confidence minimal ini.
# Lebih rendah dari CONF_OK supaya tidak berganti hijau dan merah di sekitar ambang, tapi tidak serendah
# CONF_FLOOR supaya tebakan lemah (rambut, bayangan) tidak menahan barang yang sudah diangkat.
CONF_KEEP = 0.5

# Mencari benda di meja dengan membandingkan frame ke gambar meja kosong (app/checks.py)
OBJECT_BLUR_KERNEL = 7
GRADIENT_BLUR_KERNEL = 5
# Beda warna (jarak di ruang Lab) di atas angka ini dianggap benda.
OBJECT_CHROMA_THRESHOLD = 14
# Bobot kecerahan dibuat kecil supaya bayangan dan pantulan tidak dianggap benda.
OBJECT_L_WEIGHT = 0.15
# Beda kekuatan tepi di atas angka ini dianggap benda. Ini yang menangkap botol bening.
OBJECT_GRADIENT_THRESHOLD = 20
# Opening besar membuang garis tipis (misalnya nat keramik yang bergeser) tapi tidak membuang barang.
OBJECT_OPEN_KERNEL = 15
OBJECT_CLOSE_KERNEL = 35
# Blob lebih kecil dari ini (piksel) atau lebih tipis dari MIN_BLOB_SIDE dibuang sebagai noise.
MIN_BLOB_AREA = 1500
# Batas yang sama untuk cek benda di aplikasi dan evaluate.py (find_object_blobs). Dipisah dari MIN_BLOB_AREA
# karena auto_label.py ikut memakai MIN_BLOB_AREA, dan menaikkannya membuat 16 foto produk kehilangan label.
# Produk asli terkecil di data latih seluas sekitar 3.300 piksel, jadi 3000 aman. Di atas 5000 cek benda mulai
# melewatkan benda asing di foto tes.
MIN_LIVE_BLOB_AREA = 3000
MIN_BLOB_SIDE = 30
# Daerah (x, y, lebar, tinggi) yang diabaikan karena ada benda tetap atau kabel di pojok frame.
# Isinya khusus untuk posisi kamera kita. Sesuaikan kalau kamera dipindah.
OBJECT_IGNORE_REGIONS = ((520, 0, 120, 100), (0, 0, 90, 130))

# Label otomatis foto produk (tools/auto_label.py)
# Meja kosong dibuat dari median sekian foto yang dipilih merata dari semua foto produk.
BACKGROUND_MAX_FRAMES = 60
# Saat menghitung ulang meja kosong, piksel di sekitar benda dibuang sebesar kernel ini.
BACKGROUND_EXCLUDE_KERNEL = 31
# Foto dilewati kalau ada benda kedua yang luasnya minimal segini dari benda terbesar.
AMBIGUOUS_RATIO = 0.5

# Label adegan dengan klik (tools/click_label.py)
# FastSAM dipakai hanya untuk menyiapkan data. Aplikasi tidak memakainya.
SAM_IMG_SIZE = 640
# Klik yang menghasilkan kotak seluas lebih dari ini (bagian dari foto) dianggap meleset ke lantai.
CLICK_MAX_AREA = 0.5
# Tarik mouse lebih jauh dari ini (piksel layar) berarti menggambar kotak sendiri, bukan klik.
CLICK_DRAG_MIN = 10
# Foto diperbesar sekian kali di jendela supaya mudah diklik.
CLICK_SCALE = 1.4

# Pengambilan foto (tools/capture.py)
# Jarak waktu antar foto di mode burst, dalam detik.
BURST_INTERVAL = 1.0
JPEG_QUALITY = 95

# Evaluasi (evaluation/evaluate.py)
# Ambang confidence yang dicoba. Tidak boleh di bawah CONF_FLOOR karena kotak di bawahnya sudah dibuang.
EVAL_THRESHOLDS = (0.4, 0.5, 0.65, 0.7, 0.85)
EVAL_RESULTS_DIR = ROOT / "evaluation" / "results"

# Tampilan
WINDOW_TITLE = "Self-Checkout"
# Gambar kamera diperbesar sekian kali saat ditampilkan supaya kotak dan tulisan terbaca.
DISPLAY_SCALE = 1.5
PANEL_WIDTH = 420
PANEL_MARGIN = 16
LINE_HEIGHT = 30
FONT_SCALE = 0.6
FONT_SCALE_LARGE = 0.9
TEXT_THICKNESS = 1
BOX_THICKNESS = 2
FPS_SMOOTHING = 0.9
# Satu warna per produk (BGR), urutannya sama dengan baris di products.csv. Merah dipakai untuk masalah.
CLASS_COLORS = (
    (255, 150, 0),
    (0, 220, 255),
    (220, 0, 220),
    (0, 200, 0),
    (0, 140, 255),
    (255, 255, 0),
    (180, 105, 255),
    (255, 0, 120),
)
COLOR_PROBLEM = (0, 0, 230)
# Merah lebih terang untuk teks alasan kunci di panel gelap. COLOR_PROBLEM terlalu gelap untuk dibaca.
COLOR_REASON = (90, 90, 255)
# Kotak benda di meja yang tidak dijelaskan kotak produk mana pun (yang membuat kunci tidak dikenali).
COLOR_BLOB = (255, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (0, 200, 255)
COLOR_PANEL = (40, 40, 40)
COLOR_LABEL_TEXT = (255, 255, 255)

# Penstabilan deteksi antar frame (app/stabilize.py)
# Barang baru dihitung setelah terlihat sekian frame berturut-turut.
STABLE_FRAMES = 5
# Barang dibuang dari daftar setelah hilang sekian frame berturut-turut.
LOST_FRAMES = 10
# Jarak maksimal (piksel) titik tengah kotak supaya dianggap barang yang sama antar frame.
MATCH_MAX_DIST = 60

# Kesehatan kamera (app/camera.py)
# Rata-rata kecerahan (kanal V, HSV) di bawah ini dianggap gelap.
CAMERA_DARK_V = 40
# Simpangan baku kecerahan di bawah ini dianggap tertutup (gambar hampir tanpa tekstur).
CAMERA_BLOCKED_STD = 5

# Kunci transaksi (app/lock.py, app/checks.py, main.py)
# Meja harus diam dan tidak ada masalah selama ini sebelum status jadi SIAP.
STILL_SECONDS = 1.5
# Benda asing, tepi, dan gerak baru dianggap masalah setelah aktif sekian frame berturut-turut. Bayangan,
# pantulan, atau tangan yang lewat sesaat tidak lagi langsung mengunci. Benda asli yang diam tetap terdeteksi.
LOCK_CONFIRM_FRAMES = 8
# Cek benda asing dan tepi lewat selisih ke gambar meja kosong (tombol K). Matikan (False) kalau di tempat
# demo cahaya, bayangan, atau dinding membuatnya mengunci terus. Kalau mati, tombol K tidak diperlukan dan
# benda yang tidak diberi kotak oleh YOLO tidak lagi menghalangi tombol Selesai.
CHECK_OBJECT_BLOBS = True
# Jarak (piksel) dari pinggir frame yang membuat benda TIDAK DIKENAL mendapat pesan "tepi" yang lebih
# spesifik, bukannya pesan "tidak dikenal" biasa. Hanya berlaku untuk benda yang tidak dikenal, karena
# TABLE_REGION dipotong pas dengan area taruh barang, jadi produk yang dikenal YOLO pun sering
# kotaknya menyentuh pinggir potongan gambar meski diletakkan wajar. Angka ini tidak mengubah kapan
# sistem mengunci, hanya pesannya.
EDGE_MARGIN = 20
# Beda kecerahan per piksel di atas ini dianggap berubah, untuk cek gerakan.
MOTION_PIXEL_DIFF = 15
# Bagian piksel yang berubah di atas ini dianggap meja sedang bergerak.
MOTION_MIN = 0.02
# Bagian area benda yang harus tertutup kotak produk dikenal, di bawah ini dianggap benda asing.
OBJECT_COVER_MIN = 0.6
# Jumlah frame yang dirata-ratakan saat tombol K ditekan, jadi gambar referensi meja kosong.
BACKGROUND_CAPTURE_FRAMES = 15

# Tombol Selesai (app/display.py)
BUTTON_WIDTH = 220
BUTTON_HEIGHT = 50
COLOR_PANEL_READY = (0, 170, 0)
COLOR_PANEL_LOCKED = (90, 90, 90)

# Log transaksi (app/logger.py)
LOGS_DIR = ROOT / "logs"
