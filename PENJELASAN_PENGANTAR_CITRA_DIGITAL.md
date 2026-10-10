# Pengantar Citra Digital
*(Penjelasan disusun berdasarkan program `program_pcd.py` dan hasil outputnya di folder ini)*

---

## 1. Apa itu Citra Digital?

Citra (image) adalah gambaran dua dimensi yang merupakan fungsi intensitas cahaya **f(x, y)**. Agar dapat diproses komputer, citra harus diubah menjadi **citra digital** melalui dua langkah:

| Langkah | Artinya | Di program ini |
|---|---|---|
| **Sampling (pemindaian)** | Membagi citra menjadi grid titik-titik kecil yang disebut **piksel** — menentukan *resolusi spasial* (mis. 400×400 piksel) | `baca_citra()` membaca file menjadi array `H x W` |
| **Kuantisasi** | Membulatkan nilai intensitas tiap piksel ke level-level diskret, umumnya **0–255 (8 bit)** | `dtype=np.uint8` pada semua matriks |

Satu piksel = satu elemen matriks. Nilai piksel = intensitas cahaya pada titik itu (0 = hitam total, 255 = putih total).

---

## 2. Representasi Citra sebagai Matriks

Ini inti dari program: citra **adalah** matriks numerik.

- **Citra biner** → matriks `H x W`, isi hanya `0` (hitam) atau `1` (putih).
- **Citra grayscale** → matriks `H x W`, isi `0–255` (1 kanal, 8 bit/piksel).
- **Citra RGB (berwarna)** → matriks `H x W x 3`, tiga kanal (Red, Green, Blue), masing-masing `0–255` → 24 bit/piksel.

Contoh nyata dari `hasil_grayscale.csv` (baris pertama, 8 piksel pertama):

```
206, 207, 207, 208, 208, 208, 207, 207   → area terang (hampir putih)
```

dan `hasil_biner.csv` pada posisi yang sama:

```
1, 1, 1, 1, 1, 1, 1, 1   → karena 206–208 > ambang 145 → putih
```

Semua matriks lengkap disimpan ke `hasil_R.csv`, `hasil_G.csv`, `hasil_B.csv`, `hasil_grayscale.csv`, `hasil_biner.csv` (ukuran citra contoh: **400 × 400 piksel**).

### Kebutuhan memori (citra 400×400)
| Representasi | Hitungan | Ukuran |
|---|---|---|
| RGB 8-bit | 400 × 400 × 3 kanal × 1 byte | ± 469 KB |
| Grayscale 8-bit | 400 × 400 × 1 byte | ± 156 KB |
| Biner | 400 × 400 × 1 bit (di program disimpan sbg uint8 → 1 byte) | ± 19,5 KB (teoritis) |

---

## 3. Tiga Level Citra dalam Program

### a) RGB (8 bit) — `baca_citra()`
```python
img = Image.open(path).convert("RGB")
return np.array(img, dtype=np.uint8)   # H x W x 3
```
Model warna RGB: setiap warna = kombinasi aditif tiga kanal. Contoh piksel pertama di CSV: R=209, G≈208, B≈206 → ketiganya hampir sama → warna nyaris abu-abu terang.

### b) Grayscale — `ke_grayscale()`
```
Gray = 0.299·R + 0.587·G + 0.114·B
```
Bobot bukan asal pilih — mengikuti **kepekaan mata manusia**: mata paling sensitif hijau (0.587), lalu merah (0.299), paling lemah biru (0.114). Ini standar ITU-R BT.601. Konversi grayscale = **kompresi informasi**: dari 3 kanal menjadi 1 kanal.

### c) Biner — `ke_biner()` + `hitung_otsu()`
Thresholding: `piksel > ambang → 1 (putih), selainnya → 0 (hitam)`.

Program memakai **metode Otsu**: ambang dipilih *otomatis* dengan mencari nilai T yang memaksimalkan **variansi antar-klas** antara kelompok gelap dan terang — artinya pemisahan gelap/terang paling "adil" berdasarkan histogram citra. Pada citra contoh, Otsu menghasilkan **ambang = 145**, dan hasilnya 111.220 dari 160.000 piksel (± 69,5%) bernilai putih.

---

## 4. Alur Pemrosesan (Pipeline)

```
File gambar (bmp/jpg/png)
        │  baca_citra()          ← sampling, decoding
        ▼
Matriks RGB  H×W×3 (uint8)
        │  ke_grayscale()        ← 3 kanal → 1 kanal (luminance)
        ▼
Matriks Grayscale H×W (0–255)
        │  hitung_otsu() → ambang
        │  ke_biner()            ← kuantisasi ekstrem: 256 level → 2 level
        ▼
Matriks Biner H×W (0/1)
        │
        ├─ tampilkan_matriks()   ← cetak potongan 8×8 ke console
        ├─ simpan_matriks()      ← ekspor CSV (matriks penuh)
        └─ tampilkan()           ← visualisasi RGB | Grayscale | Biner berdampingan
```

Perhatikan `interpolation="nearest"` pada `tampilkan()`: tiap piksel citra digambar 1:1 tanpa penghalusan, supaya yang dilihat benar-benar nilai matriks aslinya — sesuai prinsip bahwa **citra digital = diskrit**.

---

## 5. Kesimpulan

Dari program ini dapat disimpulkan konsep dasar pengantar citra digital:

1. Citra digital adalah **matriks angka** hasil sampling (resolusi spasial) dan kuantisasi (level abu-abu).
2. Kedalaman bit menentukan jumlah level: biner = 1 bit (2 level), grayscale = 8 bit (256 level), RGB = 24 bit (16,7 juta warna).
3. Konversi antar-representasi adalah **operasi matriks**: grayscale = kombinasi terbobot kanal RGB; biner = perbandingan dengan ambang (Otsu).
4. Tidak ada operasi resize pada program → resolusi spasial citra keluaran identik dengan input, hanya level intensitasnya yang berubah.

Konsep-konsep ini (piksel, matriks, histogram, thresholding) adalah fondasi sebelum masuk ke materi pengolahan citra tingkat lanjut: perbaikan kualitas (enhancement), filter spasial, morfologi, segmentasi, dan ekstraksi fitur.
