# Sistem Checkout Mandiri Berbasis Computer Vision

Aplikasi desktop Python yang mengenali barang belanja di atas meja lewat satu webcam, lalu menampilkan daftar belanja dan total harganya.

## Deskripsi proyek

Kasir sering menjadi bagian paling lambat dalam berbelanja. Proyek ini mencoba checkout mandiri: pelanggan menaruh semua barang di area meja, lalu aplikasi mengenali tiap barang dan menghitung totalnya, tanpa memindai satu per satu. Tujuan akhirnya, transaksi hanya boleh selesai kalau semua benda di meja sudah terhitung.

Cara kerjanya:
1. Webcam yang dipasang tetap merekam area meja, dan OpenCV memotongnya ke area yang ditentukan.
2. Model YOLO yang dilatih ulang pada 8 produk mengenali tiap barang beserta posisinya.
3. Kotak kembar pada satu benda dibuang (yang confidence-nya lebih rendah). Barang dengan confidence di atas ambang masuk daftar belanja dan dijumlahkan. Barang dengan confidence rendah diberi kotak merah "Tidak dikenal" dan tidak ikut dihitung.
4. Deteksi antar frame distabilkan (app/stabilize.py), dan OpenCV memeriksa meja langsung dari gambar (bukan dari YOLO): benda asing, barang di tepi area, dan gerakan (app/checks.py).
5. Kalau salah satu kondisi di atas aktif, atau kamera bermasalah, tombol Selesai terkunci (digambar abu-abu dengan ikon gembok). Panel menjelaskan semua penyebabnya satu per satu, lengkap dengan apa yang harus dilakukan pembeli. Daftar belanja dan total tetap tampil. Kunci terbuka sendiri begitu masalahnya hilang.
6. Transaksi dicatat ke CSV dan struk sederhana dicetak ke terminal saat tombol Selesai ditekan.

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
| Daftar belanja dan total otomatis (total selalu tampil) | Sudah |
| Penstabilan hasil antar frame | Sudah |
| Kunci transaksi otomatis dan tombol Selesai (dengan gembok dan alasan) | Sudah |
| Pemeriksaan benda asing dan barang di tepi area meja | Sudah |
| Pemeriksaan gerakan dan kesehatan kamera | Sudah |
| Pemeriksaan barang bertumpuk | Belum, sengaja tidak dibuat |
| Log transaksi CSV dan struk sederhana di terminal | Sudah |
| Evaluasi pengenalan dan kunci pada 38 foto tes | Sudah |
| Alat pengumpulan dan pelabelan data | Sudah |

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
│   ├── camera.py            membuka kamera, memotong area meja, mengubah ukuran, cek kesehatan kamera
│   ├── detector.py          menjalankan YOLO pada satu frame
│   ├── display.py           menggambar kotak, daftar belanja, total, gembok, dan tombol Selesai
│   ├── checks.py            mencari benda di meja dengan selisih gambar: benda asing, tepi, gerakan
│   ├── stabilize.py         menstabilkan deteksi antar frame, tanpa library tracker
│   ├── lock.py              state machine kunci transaksi
│   ├── logger.py            mencatat transaksi ke CSV dan mencetak struk
│   └── products.py          membaca products.csv
├── tools/
│   ├── set_table_region.py  memilih area meja dengan menarik kotak di layar
│   ├── capture.py           mengambil foto produk, latar, dan adegan
│   ├── auto_label.py        label kotak otomatis untuk foto satu produk
│   └── click_label.py       label foto adegan dengan klik (memakai FastSAM)
├── evaluation/
│   ├── evaluate.py          mengukur pengenalan barang dan kunci transaksi pada foto tes
│   └── results/             hasil evaluasi (CSV dan grafik)
├── logs/                    transactions.csv dari app/logger.py (tidak masuk git)
└── samples/                 38 foto tes dan scenes.csv (daftar barang dan should_lock yang benar)
```

## Kebutuhan perangkat lunak dan dependensi

**Perangkat:** laptop atau komputer dengan webcam. Tidak perlu GPU.

**Sistem operasi:** macOS atau Windows.

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

Jendela aplikasi terbuka. Tombol yang tersedia:

| Tombol | Fungsi |
|---|---|
| K | Simpan gambar meja kosong sebagai referensi. Pastikan meja benar-benar kosong dulu. Tekan ulang kalau cahaya atau posisi kamera berubah |
| Enter, atau klik tombol Selesai | Catat transaksi dan cetak struk di terminal. Hanya aktif saat status SIAP |
| Q atau Esc | Keluar |

Tombol Selesai digambar abu-abu dengan ikon gembok dan satu baris alasan selama terkunci (kamera bermasalah, tombol K belum ditekan, ada benda asing, barang di tepi meja, atau meja belum diam), dan hijau saat aktif.

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

Opsi `--model` memilih file model, dan `--conf` memilih ambang confidence (bawaannya `CONF_OK` di `config.py`). Kalau folder `data/product_photos/` ada (dataset lengkap, lihat bagian model dan konfigurasi tambahan), evaluasi kunci transaksi (`kunci.csv`) ikut dibuat.

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
3. Tekan K saat meja benar-benar kosong, supaya cek benda asing dan tepi bisa aktif. Kalau di tempat demo cahaya atau bayangan membuat aplikasi terkunci terus, matikan cek ini dengan `CHECK_OBJECT_BLOBS = False` di `config.py`. Tombol K lalu tidak diperlukan, tapi benda yang tidak diberi kotak oleh YOLO tidak lagi menghalangi tombol Selesai.
4. Letakkan 1 sampai 5 barang di area itu. Jangan bertumpuk (belum dicek), dan jangan terlalu dekat pinggir area.

**Evaluasi (`evaluation/evaluate.py`):** inputnya foto di folder `samples/` dan file `samples/scenes.csv`. Satu baris per foto dengan kolom:

| Kolom | Isi |
|---|---|
| `file` | Nama foto |
| `items` | Daftar barang yang benar, contoh `aqua_600ml:2\|kahf_revered:1` (nama produk lalu jumlah, dipisah garis tegak) |
| `category` | `normal`, `mirip`, `cahaya`, `asing`, atau `tumpuk` |
| `should_lock` | `yes` kalau sistem seharusnya menahan transaksi, selain itu `no`. Dipakai `evaluate.py` untuk mengukur cek benda asing dan tepi (lihat `kunci.csv`). Cek gerakan dan kesehatan kamera butuh dua frame berurutan, tidak bisa diukur dari foto diam |

## Output yang diharapkan

**Aplikasi.** Satu jendela dengan dua bagian. Di kiri, gambar kamera yang dipotong ke area meja, dengan kotak berwarna di setiap barang yang dikenali beserta nama produk dan confidence-nya (tiap produk punya warna sendiri). Kotak merah tipis bertuliskan "Tidak dikenal" menandai benda dengan confidence di bawah `CONF_OK`, dan benda itu tidak dihitung. Di kanan, panel berisi FPS, status, daftar belanja (jumlah, nama, subtotal), total harga, dan tombol Selesai. Contoh panel untuk satu Cussons dan satu Walens, status SIAP:

```text
Self-Checkout
FPS: 17.3
Status: SIAP

Daftar belanja
1 x Cussons Baby Wipes
      Rp 20.000
1 x Walens Blue Berry
      Rp 12.000

TOTAL Rp 32.000

[ SELESAI ]
K: referensi meja kosong   Q: keluar
```

Saat terkunci, tombol berubah abu-abu dengan ikon gembok, dan blok "Kenapa terkunci" muncul di atasnya. Setiap penyebab ditulis satu per satu dengan tindakannya, misalnya:

```text
Kenapa terkunci
Ada 2 benda yang tidak dikenali
  Lakukan: Angkat benda selain produk yang dijual
Ada 1 benda tidak dikenali di pinggir meja
  Lakukan: Geser ke tengah atau angkat dari meja
```

Penyebab yang mungkin: kamera terlalu gelap atau tertutup atau macet, tombol K belum ditekan, ada benda yang tidak dikenali, benda tidak dikenali di pinggir meja, dan meja sedang bergerak. Saat tidak ada masalah, panel menampilkan petunjuk singkat (misalnya "Semua barang dikenali. Tekan Selesai"). Baris "Kotak ragu-ragu (merah): jumlahnya" di bawah daftar menghitung kotak YOLO dengan confidence rendah, dan berbeda dari benda tidak dikenali di meja. Benda yang dihitung sebagai "tidak dikenali di meja" ditandai kotak putih bertuliskan "benda?" di gambar kamera, jadi terlihat jelas apa yang memicu kunci. Alasan "Meja sedang bergerak" menyebut berapa persen gambar yang berubah, untuk dibandingkan dengan `MOTION_MIN`. Angka FPS di contoh hanya ilustrasi.

Saat tombol Selesai ditekan, struk sederhana dicetak ke terminal:

```text
===== STRUK =====
25 Okt 2026  20:03
1 x Cussons Baby Wipes          Rp 20.000
1 x Walens Blue Berry           Rp 12.000
-------------------------------
TOTAL                      Rp 32.000
===============================
```

**Evaluasi.** Ringkasan di terminal, misalnya:

```text
Kunci: benar terkunci 0.667, salah kunci 0.13
Ambang confidence: 0.65, 38 adegan
Precision 0.957, recall 0.848
Daftar persis benar 0.526, total harga tepat 0.526, selisih rata-rata Rp 10318
```

Hasil lengkap ditulis ke `evaluation/results/<tanggal>_<nama model>/`:

| File | Isi |
|---|---|
| `per_produk.csv` | Benar, salah ada, terlewat, precision, dan recall tiap produk |
| `per_adegan.csv` | Daftar benar dan daftar tebakan tiap foto, dan selisih total harga |
| `per_kategori.csv` | Ringkasan per kategori |
| `ambang.csv` | Hasil pada confidence 0,4, 0,5, 0,65, 0,7, dan 0,85 |
| `confusion_matrix.csv` dan `.png` | Produk yang tertukar, barang yang tidak terdeteksi, dan benda ekstra |
| `kunci.csv` | Per foto: kategori, `should_lock`, dan apakah benda asing atau tepi terdeteksi. Hanya dibuat kalau `data/product_photos/` ada |

Hasil model yang ada di repo pada confidence 0,65 (38 foto tes):

| Kategori | Adegan | Precision | Recall | Daftar persis benar |
|---|---|---|---|---|
| Semua | 38 | 0,957 | 0,848 | 52,6% |
| normal | 15 | 0,980 | 0,906 | 66,7% |
| mirip | 8 | 1,000 | 0,833 | 62,5% |
| asing | 4 | 0,938 | 0,938 | 50,0% |
| cahaya | 6 | 0,909 | 0,909 | 50,0% |
| tumpuk | 5 | 0,933 | 0,609 | 0,0% |

Kunci (benda asing dan tepi saja, gerakan dan kamera tidak bisa diuji dari foto diam): benar terkunci 66,7% (foto kategori asing dan tumpuk yang seharusnya terkunci), salah kunci 13% (foto normal dan mirip yang seharusnya tidak terkunci).

Perbandingan pengenalan berdasarkan jumlah barang per produk, bukan posisi kotak, karena foto tes hanya punya daftar barang yang benar.

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
| `CONF_FLOOR` | Kotak dengan confidence di bawah ini dibuang | 0,4 |
| `OVERLAP_IOU` | Dua kotak yang saling menimpa lebih dari ini dianggap benda yang sama, yang lebih lemah dibuang | 0,5 |
| `CONF_OK` | Confidence minimal supaya barang baru dihitung dikenali | 0,65 |
| `CONF_KEEP` | Barang yang sudah dikenali bertahan selama confidence-nya minimal ini, supaya tidak berganti hijau dan merah di sekitar `CONF_OK` | 0,5 |
| `STABLE_FRAMES`, `LOST_FRAMES`, `MATCH_MAX_DIST` | Penstabilan deteksi antar frame | 5, 10, 60 |
| `STILL_SECONDS` | Meja harus diam sekian detik sebelum status SIAP | 1,5 |
| `LOCK_CONFIRM_FRAMES` | Benda asing, tepi, dan gerak baru dianggap masalah setelah aktif sekian frame berturut-turut, supaya bayangan atau tangan yang lewat sesaat tidak langsung mengunci | 8 |
| `CHECK_OBJECT_BLOBS` | Menyalakan cek benda asing dan tepi (tombol K). Matikan kalau cahaya di tempat demo membuatnya mengunci terus | True |
| `OBJECT_COVER_MIN` | Bagian area benda yang harus tertutup kotak dikenal, di bawahnya dianggap benda asing | 0,6 |
| `EDGE_MARGIN` | Jarak pinggir frame untuk pesan "tepi" (bukan kapan mengunci, lihat keterbatasan) | 20 |
| `DISPLAY_SCALE` | Pembesaran gambar di jendela | 1,5 |

## Keterbatasan aplikasi yang diketahui

1. **Cek barang bertumpuk tidak dibuat.** Barang yang tertutup penuh oleh barang lain tidak bisa dilihat dari satu kamera di atas, dan tidak ada cek khusus untuk itu. Pada foto tes kategori tumpuk, daftar persis benar turun ke 0% (confidence 0,65).
2. **Fitur panggil petugas tidak dibuat.** Kalau masalah tidak selesai, tombol Selesai tetap terkunci tanpa batas waktu.
3. **Cek benda asing dan gerakan bisa mengunci terus di tempat dengan cahaya, bayangan, atau dinding yang berbeda dari data latih.** Dalam percobaan manual di kamera lain, aplikasi sering terkunci padahal meja kosong. Penyebab pastinya belum dibuktikan, tapi percobaan memperlihatkan kotak "benda?" muncul di petak lantai kosong, di pojok dinding, dan di pinggir gambar, dan berpindah-pindah. Karena itu kondisi ini baru dianggap masalah setelah aktif beberapa frame berturut-turut (`LOCK_CONFIRM_FRAMES`). Dugaan: `TABLE_REGION` belum diatur sehingga dinding dan tepi lantai ikut terbaca (jalankan `tools/set_table_region.py`), gambar meja kosong dari tombol K sudah usang karena cahaya berubah (tekan K lagi), atau bayangan botol ikut terhitung sebagai benda. Kotak putih "benda?" dan persen gerak di layar dipakai untuk melihat penyebabnya, dan `CHECK_OBJECT_BLOBS = False` mematikan cek ini.
4. **Cek gerakan dan kesehatan kamera belum diukur dengan angka.** Keduanya butuh dua frame berurutan, tidak bisa diuji dari foto tes yang diam, jadi baru dites manual di depan kamera, hasilnya belum dicatat formal.
5. **`EDGE_MARGIN` hanya memengaruhi pesan, bukan kapan mengunci.** Diukur: kalau dipakai untuk mengunci langsung (bukan hanya pesan), 20 dari 23 foto normal ikut salah terkunci, karena `TABLE_REGION` dipotong pas dengan area taruh barang. Sekarang cek tepi hanya berlaku untuk benda yang juga tidak dikenal YOLO, supaya produk sah yang mepet potongan frame tidak ikut terkunci.
6. **Benda yang mirip produk bisa dikenali sebagai produk.** Pada pengujian, HP hitam dikenali sebagai Kahf dengan confidence 0,46, dan tangan dikenali sebagai Walens atau Vaseline. Cek benda asing (`unknown_objects`) hanya menangkap benda yang TIDAK diberi kotak YOLO sama sekali, bukan benda yang diberi kotak dengan nama salah.
7. **Botol bening dan kemasan mirip.** Warna botol bening hampir sama dengan lantai, dan Le Minerale paling sering terlewat. Pada foto validasi training, sekitar 20 dari 45 foto Le Minerale dikenali sebagai Aqua.
8. **Cahaya redup menurunkan hasil.**
9. **Hanya 8 produk.** Produk lain tidak dikenali dan bisa salah ditebak sebagai salah satu dari 8 produk itu.
10. **Lingkungan pelatihan terbatas.** Foto diambil di satu ruangan dengan lantai keramik krem yang mengkilap, dengan satu kamera. Hasil di tempat lain belum diketahui.
11. **Data kecil.** 936 foto produk, 112 foto adegan berlabel, dan 38 foto tes. Angka evaluasi bukan jaminan untuk kondisi lain.
12. **Pengukuran berdasarkan jumlah barang.** Evaluasi tidak memeriksa posisi kotak.
13. **Kecepatan.** FPS di pojok layar terbaca sekitar 15 sampai 25 di laptop tanpa GPU, tapi belum diukur dengan prosedur yang rapi, dan bisa berbeda di laptop lain.
14. **Model tetap `yolo26n`.** `yolo26s` sudah dicoba dan sekitar 2,2 kali lebih lambat saat deteksi, berisiko membuat FPS turun di bawah 5, jadi tidak dipakai.
15. **Cek benda asing memakai selisih warna dan tepi**, bukan model deteksi siap pakai seperti FastSAM. FastSAM otomatis sudah dicoba dan sekitar 25 kali lebih lambat, tidak muat di target 5 FPS minimum.
16. **Alas yang berbeda dari data latih kemungkinan menurunkan hasil.** Dalam percobaan manual di alas abu-abu mengkilap, Cussons pernah terbaca Aqua dengan confidence 0,73. Penyebabnya belum dibuktikan dan belum diukur dengan angka. Foto latih semuanya di lantai keramik krem.
17. **Mekanisme kunci transaksi diuji dengan kamera tiruan** (membaca foto berulang, bukan webcam asli), supaya alurnya bisa dites tanpa menunggu kondisi tertentu terjadi di depan kamera. Belum diuji dengan webcam sungguhan.
18. **`TABLE_REGION` dan `OBJECT_IGNORE_REGIONS` di `config.py`** mengikuti posisi kamera kami. Pada kamera lain, area meja perlu dipilih ulang.

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
