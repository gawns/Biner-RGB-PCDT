# Program Pengolahan Citra Digital

Program Python untuk membaca citra dan menampilkannya dalam tiga representasi: **biner**, **grayscale (8 bit)**, dan **RGB (8 bit)**, serta merepresentasikannya sebagai **matriks** — dilengkapi **gambar hasil pemrosesan (PNG)** dan **laporan otomatis (Markdown)** berisi cara sistem berjalan serta status aman/tidaknya.

## Spesifikasi

| No | Kebutuhan | Penyelesaian |
|----|-----------|--------------|
| 1 | Input gambar bertipe `bmp`, `jpg`, `png` | Validasi ekstensi + `Pillow` |
| 2 | Tampil sebagai biner, grayscale (8 bit), RGB (8 bit) | Konversi dengan `numpy` |
| 3 | Resolusi sesuai input | Tidak ada proses *resize* |
| 4 | Dapat direpresentasikan dalam matriks | Array `numpy` (`H x W` atau `H x W x 3`) |
| 5 | Gambar hasil pemrosesan | Ekspor PNG lossless (RGB, R, G, B, grayscale, biner) di `hasil_pemrosesan/` |
| 6 | Laporan otomatis | `hasil_pemrosesan/laporan_pemrosesan.md` — pipeline, statistik, dan pemeriksaan keamanan |
| 7 | Sistem berjalan aman / tidak | Status akhir `AMAN — SUKSES` / `SELESAI DENGAN PERINGATAN` / `GAGAL` + 13 pemeriksaan otomatis |

## Instalasi Library

```bash
pip install pillow numpy matplotlib
```

## Menjalankan Program

```bash
python program_pcd.py                          # diminta memasukkan path gambar
python program_pcd.py <path_gambar>            # langsung memproses
python program_pcd.py <path_gambar> --no-gui   # tanpa jendela pratinjau matplotlib
python program_pcd.py <path_gambar> --folder <dir>   # folder output lain
```

## Output yang Dihasilkan

Semua output masuk folder `hasil_pemrosesan/`:

- **CSV matriks penuh**: `hasil_R.csv`, `hasil_G.csv`, `hasil_B.csv`, `hasil_grayscale.csv`, `hasil_biner.csv`
- **PNG gambar hasil**: `hasil_rgb.png`, `hasil_R.png`, `hasil_G.png`, `hasil_B.png`, `hasil_grayscale.png`, `hasil_biner.png`
- **Laporan**: `laporan_pemrosesan.md`

## Pemeriksaan Keamanan Otomatis

Program memeriksa: library terpasang, file ditemukan, format didukung, batas ukuran file (≤ 20 MB) dan resolusi (≤ 8000 px sisi / ≤ 30 juta piksel), validitas matriks grayscale & biner, kelengkapan file output, verifikasi *readback* PNG terhadap matriks (bit-per-bit), serta error runtime. Hasilnya dirangkum pada status akhir di console dan di laporan.
