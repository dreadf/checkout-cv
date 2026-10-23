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
# Kotak dengan confidence di bawah angka ini dibuang. Dibuat rendah supaya kotak ragu-ragu tetap muncul.
CONF_FLOOR = 0.25
# Confidence di atas atau sama dengan angka ini berarti "dikenali". Di bawahnya "tidak dikenal" dan tidak dihitung.
# Ini angka awal, belum dituning.
CONF_OK = 0.5

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
EVAL_THRESHOLDS = (0.25, 0.5, 0.7, 0.85)
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
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (0, 200, 255)
COLOR_PANEL = (40, 40, 40)
COLOR_LABEL_TEXT = (255, 255, 255)
