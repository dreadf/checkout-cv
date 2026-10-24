# Sistem Checkout Mandiri Berbasis Computer Vision

Aplikasi desktop Python yang mengenali barang belanja di atas meja lewat satu webcam, lalu menampilkan daftar belanja dan total harganya.

## Deskripsi proyek

Kasir sering menjadi bagian paling lambat dalam berbelanja. Proyek ini mencoba checkout mandiri: pelanggan menaruh semua barang di area meja, lalu aplikasi mengenali tiap barang dan menghitung totalnya, tanpa memindai satu per satu. Tujuan akhirnya, transaksi hanya boleh selesai kalau semua benda di meja sudah terhitung.

Cara kerjanya:
1. Webcam yang dipasang tetap merekam area meja, dan OpenCV memotongnya ke area yang ditentukan.
2. Model YOLO yang dilatih ulang pada 8 produk mengenali tiap barang beserta posisinya.
3. Barang dengan confidence di atas ambang masuk daftar belanja dan dijumlahkan. Barang dengan confidence rendah diberi kotak merah "Tidak dikenal" dan tidak ikut dihitung.
4. Hasilnya digambar di layar dengan OpenCV.

Produk yang dikenali:

| Produk | Harga per unit |
|---|---|
| Aqua 600ml | Rp 3.900 |
| Le Minerale 600ml | Rp 4.000 |
| Walens Blue Berry | Rp 12.000 |
| Cussons Baby Wipes | Rp 20.000 |
| Hansaplast 10 Lembar | Rp 24.000 |
| Kahf Revered | Rp 80.000 |
| Soffell Kulit Jeruk | Rp 500 |
| Vaseline UV Brightening | Rp 28.000 |

### Status fitur

| Fitur | Status |
|---|---|
| Akuisisi webcam dan pemotongan area meja | Sudah |
| Pengenalan 8 produk (YOLO26n hasil fine-tuning) | Sudah |
| Daftar belanja dan total otomatis | Sudah |
| Status "Tidak dikenal" untuk confidence di bawah 0,5 | Sudah |
| Evaluasi pengenalan pada 38 foto tes | Sudah |
| Alat pengumpulan dan pelabelan data | Sudah |
| Penstabilan hasil antar frame | Belum |
| Kunci transaksi dan tombol Selesai | Belum |
| Pemeriksaan benda asing, barang bertumpuk, dan gerakan di aplikasi | Belum. Fungsi pencari benda ada di `app/checks.py` tapi baru dipakai alat pelabelan |
| Log transaksi | Belum |

### Struktur folder

```text
checkout-cv/
├── main.py                  aplikasi utama
├── config.py                semua angka yang bisa diatur
├── products.csv             daftar produk dan harga
├── best.pt                  model hasil training (v2, dipakai aplikasi)
├── best_v1.pt               model v1, hanya pembanding
├── train_colab.ipynb        notebook training (Google Colab)
├── requirements.txt         dependensi
├── README.md
├── app/
│   ├── camera.py            membuka kamera, memotong area meja, mengubah ukuran
│   ├── detector.py          menjalankan YOLO pada satu frame
│   ├── display.py           menggambar kotak, daftar belanja, dan total
│   ├── checks.py            mencari benda di meja dengan selisih gambar (OpenCV)
│   └── products.py          membaca products.csv
├── tools/
│   ├── set_table_region.py  memilih area meja dengan menarik kotak di layar
│   ├── capture.py           mengambil foto produk, latar, dan adegan
│   ├── auto_label.py        label kotak otomatis untuk foto satu produk
│   └── click_label.py       label foto adegan dengan klik (memakai FastSAM)
├── evaluation/
│   ├── evaluate.py          mengukur pengenalan barang pada foto tes
│   └── results/             hasil evaluasi (CSV dan grafik)
└── samples/                 38 foto tes dan scenes.csv (daftar barang yang benar)
```

## Kebutuhan perangkat lunak dan dependensi

**Perangkat:** laptop atau komputer dengan webcam. Tidak perlu GPU.

**Sistem operasi:** macOS (sudah diuji) atau Windows (belum diuji).

**Python:** 3.11 atau 3.12. Pengujian memakai Python 3.12.

**Library** (versi persis ada di `requirements.txt`):

| Library | Versi | Fungsi |
|---|---|---|
| opencv-python | 4.14.0.94 | Kamera, memotong dan mengubah ukuran gambar, pencarian benda, menggambar hasil |
| numpy | 2.5.4 | Operasi array gambar |
| ultralytics | 8.4.183 | Menjalankan dan melatih model YOLO. PyTorch ikut terpasang otomatis |
| matplotlib | 3.11.2 | Grafik confusion matrix pada evaluasi |

Pemasangan pertama membutuhkan internet dan beberapa menit, karena PyTorch ukurannya besar.

## Petunjuk instalasi

```bash
git clone https://github.com/dreadf/checkout-cv.git
cd checkout-cv
```

Buat virtual environment (folder Python khusus proyek ini) lalu pasang library.

macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (Command Prompt atau PowerShell):

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Setiap membuka terminal baru, jalankan lagi perintah `activate`.

## Cara menjalankan aplikasi

```bash
python main.py
```

Jendela aplikasi terbuka. Tekan **Q** atau **Esc** untuk keluar.

Kalau kamera tidak terbuka:
1. Di macOS, beri izin kamera untuk aplikasi terminal di System Settings, Privacy and Security, Camera.
2. Ubah `CAMERA_INDEX` di `config.py` (0, 1, atau 2). Nilai 0 adalah webcam bawaan di kebanyakan laptop. Laptop kami memakai 1 karena nomor 0 diambil kamera virtual.

### Memilih area meja (disarankan)

Model dilatih dari foto yang dipotong ke area meja, jadi hasilnya paling baik kalau area itu dipilih. Bawaannya `TABLE_REGION = None`, yaitu memakai seluruh gambar kamera.

```bash
python tools/set_table_region.py
```

Tarik kotak mengikuti pinggir area tempat barang diletakkan, lalu tekan Enter. Salin baris `TABLE_REGION = ...` yang tercetak ke `config.py`. Untuk kamera dan posisi kami, nilainya `(264, 68, 768, 534)`.

### Menjalankan evaluasi

```bash
python evaluation/evaluate.py
python evaluation/evaluate.py --model best_v1.pt --conf 0.5
```

Opsi `--model` memilih file model, dan `--conf` memilih ambang confidence (bawaannya `CONF_OK` di `config.py`).

### Alat untuk menyiapkan data (opsional)

Hanya dibutuhkan kalau ingin mengumpulkan data atau melatih ulang model. Dataset lengkap ada di Google Drive (lihat bagian model dan konfigurasi tambahan) dan diletakkan di folder `data/`.

```bash
python tools/capture.py --mode product --class aqua_600ml
python tools/capture.py --mode background
python tools/capture.py --mode scene
python tools/auto_label.py
python tools/click_label.py
```

Spasi mengambil satu foto, B menyalakan burst (satu foto tiap sekitar satu detik), dan Q selesai. Di `click_label.py`, tekan angka produk (1 sampai 8) lalu klik di tengah barang. Tombol lain tertulis di panel jendelanya. Foto disimpan di `data/product_photos/<nama produk>/`, `data/background_photos/`, dan `data/scene_photos_train/`.

## Cara memberikan input

**Aplikasi (`main.py`):** inputnya video langsung dari webcam.
1. Pasang kamera tetap menghadap area meja yang rata dan cukup terang.
2. Pilih area meja (lihat di atas).
3. Letakkan 1 sampai 5 barang di area itu. Jangan bertumpuk, dan jangan terlalu dekat pinggir area.

**Evaluasi (`evaluation/evaluate.py`):** inputnya foto di folder `samples/` dan file `samples/scenes.csv`. Satu baris per foto dengan kolom:

| Kolom | Isi |
|---|---|
| `file` | Nama foto |
| `items` | Daftar barang yang benar, contoh `aqua_600ml:2\|kahf_revered:1` (nama produk lalu jumlah, dipisah garis tegak) |
| `category` | `normal`, `mirip`, `cahaya`, `asing`, atau `tumpuk` |
| `should_lock` | `yes` kalau sistem seharusnya menahan transaksi, selain itu `no`. Kolom ini belum dipakai kode mana pun karena kunci transaksi belum dibuat |

## Output yang diharapkan

**Aplikasi.** Satu jendela dengan dua bagian. Di kiri, gambar kamera yang dipotong ke area meja, dengan kotak berwarna di setiap barang yang dikenali beserta nama produk dan confidence-nya (tiap produk punya warna sendiri). Kotak merah tipis bertuliskan "Tidak dikenal" menandai benda dengan confidence di bawah 0,5, dan benda itu tidak dihitung. Di kanan, panel berisi FPS, daftar belanja (jumlah, nama, subtotal), dan total harga. Contoh panel untuk satu Cussons dan satu Walens:

```text
Self-Checkout
FPS: 17.3
Kunci transaksi belum aktif.

Daftar belanja
1 x Cussons Baby Wipes
      Rp 20.000
1 x Walens Blue Berry
      Rp 12.000

TOTAL Rp 32.000
```

Kalau ada benda yang tidak dikenal, baris "Tidak dikenal: jumlahnya" muncul di bawah daftar. Angka FPS di contoh hanya ilustrasi.

**Evaluasi.** Ringkasan di terminal, misalnya:

```text
Ambang confidence: 0.5, 38 adegan
Precision 0.929, recall 0.886
Daftar persis benar 0.553, total harga tepat 0.553, selisih rata-rata Rp 10324
```

Hasil lengkap ditulis ke `evaluation/results/<tanggal>_<nama model>/`:

| File | Isi |
|---|---|
| `per_produk.csv` | Benar, salah ada, terlewat, precision, dan recall tiap produk |
| `per_adegan.csv` | Daftar benar dan daftar tebakan tiap foto, dan selisih total harga |
| `per_kategori.csv` | Ringkasan per kategori |
| `ambang.csv` | Hasil pada confidence 0,25, 0,5, 0,7, dan 0,85 |
| `confusion_matrix.csv` dan `.png` | Produk yang tertukar, barang yang tidak terdeteksi, dan benda ekstra |

Hasil model yang ada di repo pada confidence 0,5 (38 foto tes):

| Kategori | Adegan | Precision | Recall | Daftar persis benar |
|---|---|---|---|---|
| Semua | 38 | 0,929 | 0,886 | 55,3% |
| normal | 15 | 0,926 | 0,943 | 73,3% |
| mirip | 8 | 1,000 | 0,833 | 62,5% |
| asing | 4 | 0,938 | 0,938 | 50,0% |
| cahaya | 6 | 0,875 | 0,955 | 33,3% |
| tumpuk | 5 | 0,941 | 0,696 | 20,0% |

Perbandingannya berdasarkan jumlah barang per produk, bukan posisi kotak, karena foto tes hanya punya daftar barang yang benar.

## Model atau konfigurasi tambahan yang dibutuhkan

**Model.** `best.pt` sudah ada di repo dan langsung dipakai `main.py`. Model ini YOLO26n yang dilatih ulang (fine-tuning) dengan foto produk, foto adegan, dan foto latar kami, sekitar 60 epoch pada ukuran gambar 640. Untuk melatih ulang, ikuti `train_colab.ipynb` di Google Colab dengan GPU. Fotonya dikompres jadi satu `foto.zip` berisi tiga folder (`product_photos`, `scene_photos_train`, `background_photos`) dan diunggah ke Google Drive.

**Dataset lengkap.** Foto produk, adegan, dan latar beserta label ada di Google Drive: [unduh dari Google Drive](https://drive.google.com/file/d/1VVNI-OKIuilnaiNPNye_Xqk3Etc6v4u0/view?usp=sharing). Ekstrak ke folder `data/` di repo ini.

**FastSAM.** `tools/click_label.py` memakai model FastSAM-s untuk menentukan kotak dari titik klik. Filenya (`FastSAM-s.pt`, sekitar 23 MB) diunduh otomatis saat alat itu pertama dijalankan. Model ini hanya dipakai untuk menyiapkan data, bukan oleh aplikasi.

**Pengaturan di `config.py`:**

| Nama | Arti | Bawaan |
|---|---|---|
| `CAMERA_INDEX` | Nomor kamera | 0 |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | Resolusi yang diminta dari kamera | 1280 dan 720 |
| `TABLE_REGION` | Area meja (x, y, lebar, tinggi). `None` berarti seluruh gambar | `None` |
| `IMG_SIZE` | Ukuran gambar masuk model. Harus sama dengan saat training | 640 |
| `CONF_FLOOR` | Kotak dengan confidence di bawah ini dibuang | 0,25 |
| `CONF_OK` | Confidence minimal supaya barang dihitung dikenali | 0,5 |
| `DISPLAY_SCALE` | Pembesaran gambar di jendela | 1,5 |

## Keterbatasan aplikasi yang diketahui

1. **Fitur pengaman belum ada di aplikasi.** Belum ada penstabilan antar frame, kunci transaksi, tombol Selesai, pemeriksaan benda asing, barang bertumpuk, atau gerakan, dan log transaksi. Aplikasi baru menampilkan hasil pengenalan.
2. **Benda yang mirip produk bisa dikenali sebagai produk.** Pada pengujian, HP hitam dikenali sebagai Kahf dengan confidence 0,46, dan tangan dikenali sebagai Walens atau Vaseline. Tanpa pemeriksaan benda, hal ini tidak tertangkap.
3. **Barang bertumpuk atau saling menutup.** Barang yang tertutup penuh tidak terlihat dari satu kamera. Pada foto tes kategori tumpuk, hanya 20% daftar yang persis benar.
4. **Botol bening dan kemasan mirip.** Warna botol bening hampir sama dengan lantai, dan Le Minerale paling sering terlewat (5 dari 17 pada foto tes). Pada foto validasi training, yaitu foto satu produk dari sesi yang tidak dipakai belajar, sekitar 20 dari 45 foto Le Minerale dikenali sebagai Aqua. Angka val dan tes berbeda karena fotonya memang berbeda. Di foto tes yang berisi banyak barang, Le Minerale lebih sering tidak terdeteksi daripada tertukar jadi Aqua.
5. **Cahaya redup menurunkan hasil.** Pada foto tes kategori cahaya, hanya 33% daftar yang persis benar.
6. **Hanya 8 produk.** Produk lain tidak dikenali dan bisa salah ditebak sebagai salah satu dari 8 produk itu.
7. **Lingkungan pelatihan terbatas.** Foto diambil di satu ruangan dengan lantai keramik krem yang mengkilap, dengan satu kamera. Pantulan lantai dan posisi kamera yang bergeser memengaruhi pencarian benda. Hasil di tempat lain belum diketahui.
8. **Data kecil.** 936 foto produk, 112 foto adegan berlabel, dan 38 foto tes. Angka evaluasi bukan jaminan untuk kondisi lain.
9. **Pengukuran berdasarkan jumlah barang.** Evaluasi tidak memeriksa posisi kotak.
10. **Kecepatan.** FPS di pojok layar terbaca sekitar 15 sampai 20 pada laptop kami (chip Apple M2, tanpa GPU), tapi belum diukur dengan prosedur yang rapi.
11. **Baru diuji di macOS.** Windows belum diuji.
12. **`TABLE_REGION` dan `OBJECT_IGNORE_REGIONS` di `config.py`** mengikuti posisi kamera kami. Pada kamera lain, area meja perlu dipilih ulang.

## Sumber dan lisensi

| Nama | Dipakai untuk | Lisensi |
|---|---|---|
| OpenCV | Kamera, pengolahan gambar, tampilan | Apache 2.0 |
| NumPy | Operasi array | BSD-3-Clause |
| Ultralytics, termasuk YOLO26n | Model pengenalan dan training | AGPL-3.0 |
| FastSAM (lewat Ultralytics) | Penentu kotak dari titik klik, hanya untuk pelabelan | AGPL-3.0 |
| PyTorch | Menjalankan model | BSD-3-Clause dan lisensi terbuka lain |
| Matplotlib | Grafik evaluasi | Lisensi Matplotlib (berbasis PSF) |
| Dataset COCO | Dipakai Ultralytics untuk melatih model dasar | Anotasi CC BY 4.0 |
| Google Colab | Training dengan GPU | Syarat layanan Google |

Karena memakai Ultralytics yang berlisensi AGPL-3.0, kode dalam repo ini mengikuti ketentuan lisensi yang sama kalau dibagikan.

Rujukan:
1. Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
2. Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
3. Zhao, X., et al. (2023). Fast Segment Anything. arXiv:2306.12156.
