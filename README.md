# Dashboard Kemiskinan dan Pembangunan Manusia di Indonesia

Dashboard interaktif berbasis **Streamlit** yang menampilkan kondisi kemiskinan, pengangguran, dan pembangunan manusia di Indonesia. Dashboard ini menggabungkan analisis tingkat kabupaten/kota (peta dan klaster spasial) dengan analisis tingkat provinsi (PCA dan tipologi).

> Proyek UAS Visualisasi Data. Sumber data: BPS (data diolah).

**Demo online:** _tambahkan tautan Streamlit Cloud di sini, contoh `https://uasvisdat.streamlit.app`_

## Fitur

Dashboard terdiri dari empat tab:

| Tab | Isi |
|---|---|
| **Ringkasan** | Angka kunci (jumlah provinsi, kab/kota, Moran's I, hotspot/coldspot), tabel Moran's I global, dan ukuran tiap tipologi provinsi |
| **Peta dan klaster spasial** | Peta interaktif kab/kota dengan beberapa layer (% penduduk miskin, IPM, TPT, RLS, klaster LISA, simbol proporsional), plot Moran, dan grafik share IPM |
| **Tipologi provinsi (PCA)** | Biplot, heatmap terklaster, koordinat paralel, matriks korelasi, serta loading komponen dan daftar provinsi per tipologi |
| **Bahan bakar dan internet** | Treemap bahan bakar utama memasak dan icicle akses internet rumah tangga, keduanya dengan drill-down |

## Metode singkat

- **PCA** pada 9 variabel provinsi: kesenjangan listrik PLN (log), PDRB per kapita (log), rasio penerima bansos, IPM, % memasak dengan bahan bakar bersih, % akses internet, % keluarga besar, % penduduk miskin, dan TPT. Semua variabel distandarkan terlebih dahulu.
- **K-means** (k = 4) pada tiga komponen utama pertama, menghasilkan empat tipologi provinsi:
  1. Maju dan kaya
  2. Maju, ekonomi menengah
  3. Timur tertinggal
  4. Tertinggal ekstrem
- **Uji sensitivitas** terhadap pengecualian provinsi pemekaran Papua dan DKI Jakarta, pengecualian variabel IPM, serta variabel bahan bakar bersih dan bansos.
- **Moran's I global** dan **LISA** (Local Indicators of Spatial Association) untuk mengukur autokorelasi spasial antar kab/kota, menghasilkan kategori High-High (hotspot), Low-Low (coldspot), dan lainnya.

## Variabel yang digunakan

### 1. Tingkat provinsi (38 provinsi, file `rumah_tangga.xlsx`)
Digunakan untuk visualisasi multivariat
**Variabel mentah**

| Kolom | Keterangan |
|---|---|
| `Provinsi` | Nama provinsi (baris "Indonesia" dibuang) |
| `PLN` | Persentase rumah tangga yang menggunakan listrik PLN |
| `PDRB per Capita` | Produk Domestik Regional Bruto per kapita (ribu rupiah) |
| `Bansos` | Jumlah keluarga penerima manfaat (KPM) bantuan sosial |
| `IPM` | Indeks Pembangunan Manusia |
| `Bahan Bakar Listrik` | % rumah tangga yang memasak dengan listrik |
| `Bahan Bakar Elpiji` | % rumah tangga yang memasak dengan elpiji |
| `Internet` | % rumah tangga/penduduk yang mengakses internet |
| `Keluarga Besar` | % keluarga besar |
| `PPM` | % penduduk miskin |
| `TPT` | Tingkat Pengangguran Terbuka (%) |
| `Penduduk` | Jumlah penduduk (ribu jiwa) |

**Sembilan variabel aktif dalam PCA**

Sebagian variabel mentah diubah dulu agar sebarannya lebih wajar dan arahnya mudah dibaca.

| Variabel | Cara menghitung | Arti nilai tinggi |
|---|---|---|
| `pln_gap` Kesenjangan PLN (log) | `ln(100 − PLN + 1)` | Makin tertinggal akses listrik PLN |
| `pdrb_log` PDRB/kapita (log) | `ln(PDRB per kapita dalam juta rupiah)` | Ekonomi makin kuat |
| `bansos_rasio` KPM bansos per 1.000 penduduk | `Bansos / Penduduk` | Makin banyak penerima bansos |
| `ipm` IPM | Langsung dari data | Pembangunan manusia makin tinggi |
| `bb_bersih` % masak bahan bakar bersih | `Listrik + Elpiji` | Makin banyak memakai bahan bakar bersih (LPG dan listrik) |
| `internet` % akses internet | Langsung dari data | Akses internet makin luas |
| `kel_besar` % keluarga besar | Langsung dari data | Makin banyak keluarga besar (anggota ruta >= 6)|
| `ppm` % penduduk miskin | Langsung dari data | Kemiskinan makin tinggi |
| `tpt` TPT (%) | Langsung dari data | Pengangguran makin tinggi |

Semua variabel distandarkan (z-score) sebelum PCA. Variabel `pdrb_juta` (PDRB dalam juta rupiah) juga dihitung untuk tabel ringkasan, tetapi tidak masuk PCA.

### 2. Tingkat kabupaten/kota (file `kabkota_lisa.geojson`)
Dipakai pada peta visualisasi geospasial, analisis Moran's I, dan LISA:

| Variabel | Keterangan |
|---|---|
| % penduduk miskin | Variabel utama untuk Moran's I dan LISA |
| IPM | Indeks Pembangunan Manusia |
| TPT | Tingkat Pengangguran Terbuka |
| RLS | Rata-rata Lama Sekolah |
| `lisa_kat` | Kategori LISA hasil perhitungan: High-High (hotspot), Low-Low (coldspot), dan kategori lain |

### 3. Rumah tangga (tahun 2022)
Dipakai pada visualisasi data hirarki

| Berkas | Dipakai untuk | Ukuran kotak | Warna |
|---|---|---|---|
| `bahan_bakar_2022.xlsx` | Treemap bahan bakar utama memasak | Jumlah rumah tangga | % rumah tangga yang memakai bahan bakar kotor |
| `internet_2022.xlsx` | Icicle akses internet | Jumlah rumah tangga | % rumah tangga yang pernah mengakses internet |

> Catatan: definisi resmi tiap indikator mengikuti metadata BPS. Daftar variabel kab/kota di atas diambil dari yang ditampilkan `app.py`; cocokkan dengan nama kolom pada `02_geospasial_lisa.py` jika ada yang berbeda.

## Struktur folder

```
uasvisdat/
├── app.py                    # dashboard Streamlit
├── 01_analisis_pca.py        # PCA + klaster provinsi
├── 02_geospasial_lisa.py     # peta kab/kota, LISA, Moran, treemap, icicle
├── requirements.txt
└── data/
    ├── raw/
    │   ├── rumah_tangga.xlsx         # dipakai skrip 01
    │   ├── bahan_bakar_2022.xlsx     # dipakai treemap
    │   └── internet_2022.xlsx        # dipakai icicle
    └── processed/
        ├── kabkota_lisa.geojson      # hasil skrip 02
        ├── moran_global.csv          # hasil skrip 02
        └── provinsi_terolah.csv      # hasil skrip 01
```
## Sumber data

Badan Pusat Statistik (BPS), data diolah.

## Pembuat
Evelyn Tan Eldisha Nawa
3SD2
[evelynn-tan](https://github.com/evelynn-tan)
