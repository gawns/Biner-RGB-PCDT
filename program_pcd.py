"""
Program Pengolahan Citra Digital (versi laporan otomatis)
=========================================================
Membaca citra (bmp/jpg/png) lalu memprosesnya menjadi tiga representasi:
RGB (8 bit), Grayscale (8 bit), dan Biner (0/1).

Output yang dihasilkan (di folder hasil_pemrosesan/):
  - Matriks lengkap  : hasil_*.csv
  - Gambar hasil     : hasil_*.png  (RGB, R, G, B, grayscale, biner)
  - Laporan proses   : laporan_pemrosesan.md  (cara sistem berjalan,
                       status tiap tahap, dan verdict AMAN/TIDAK AMAN)

Cara pakai:
  python program_pcd.py                      -> diminta memasukkan path
  python program_pcd.py <path_gambar>        -> langsung memproses
  python program_pcd.py <path_gambar> --no-gui   -> tanpa jendela matplotlib
  python program_pcd.py <path_gambar> --folder <dir>  -> folder output lain

Kode keluar (exit code): 0 = sukses, 1 = gagal.
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
from PIL import Image

FORMAT_DIDUKUNG = (".bmp", ".jpg", ".jpeg", ".png")

# Batas wajar untuk menjaga sistem tetap aman (menghasilkan PERINGATAN,
# bukan langsung gagal, kecuali ditentukan lain).
BATAS_UKURAN_FILE_MB = 20          # MB
BATAS_SISI_PIKSEL = 8000           # sisi terpanjang
BATAS_TOTAL_PIKSEL = 30_000_000    # total piksel

NAMA_FOLDER_OUTPUT = "hasil_pemrosesan"
NAMA_LAPORAN = "laporan_pemrosesan.md"
NAMA_DASAR = "hasil"
VERSI_PROGRAM = "2.0"


# ---------------------------------------------------------------
# 1. Membaca citra (bmp, jpg, png)
# ---------------------------------------------------------------
def baca_citra(path):
    """Membaca citra dan mengembalikan matriks RGB 8 bit (H x W x 3)."""
    ekstensi = os.path.splitext(path)[1].lower()
    if ekstensi not in FORMAT_DIDUKUNG:
        raise ValueError(f"Format '{ekstensi}' tidak didukung. Gunakan: bmp, jpg, png.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    try:
        img = Image.open(path)
        img.load()                              # paksa decoding -> file korup terdeteksi
        img = img.convert("RGB")                # menangani RGBA / palette / grayscale
    except Exception as e:
        raise ValueError(f"Gagal membaca gambar (mungkin korup/bukan gambar): {e}") from e
    return np.array(img, dtype=np.uint8)        # resolusi tetap sesuai input


# ---------------------------------------------------------------
# 2. Konversi ke grayscale 8 bit (0 - 255)
# ---------------------------------------------------------------
def ke_grayscale(rgb):
    """Gray = 0.299 R + 0.587 G + 0.114 B (standar ITU-R BT.601)."""
    gray = (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2])
    return np.round(gray).astype(np.uint8)


# ---------------------------------------------------------------
# 3. Konversi ke citra biner (0 atau 1)
# ---------------------------------------------------------------
def hitung_otsu(gray):
    """Menghitung nilai ambang (threshold) otomatis dengan metode Otsu."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    jumlah_semua = np.dot(np.arange(256), hist)

    bobot_b = 0.0
    jumlah_b = 0.0
    var_maks = 0.0
    ambang = 0

    for t in range(256):
        bobot_b += hist[t]
        if bobot_b == 0:
            continue
        bobot_f = total - bobot_b
        if bobot_f == 0:
            break
        jumlah_b += t * hist[t]
        rata_b = jumlah_b / bobot_b
        rata_f = (jumlah_semua - jumlah_b) / bobot_f
        var_antar = bobot_b * bobot_f * (rata_b - rata_f) ** 2
        if var_antar > var_maks:
            var_maks = var_antar
            ambang = t
    return ambang


def ke_biner(gray, ambang=None):
    """Piksel > ambang menjadi 1 (putih), selainnya 0 (hitam)."""
    if ambang is None:
        ambang = hitung_otsu(gray)
    return (gray > ambang).astype(np.uint8), ambang


# ---------------------------------------------------------------
# 4. Menampilkan citra dengan resolusi sesuai input (opsional/GUI)
# ---------------------------------------------------------------
def tampilkan(rgb, gray, biner):
    """Menampilkan RGB | Grayscale | Biner berdampingan.

    Mengembalikan True bila jendela grafik berhasil ditampilkan.
    Aman dipanggil di lingkungan tanpa GUI (server/terminal) -> False.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [PERINGATAN] matplotlib tidak dapat diimpor: {e}")
        return False

    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "pdf", "svg", "ps", "cairo"):
        print(f"  [PERINGATAN] Backend '{backend}' tidak mendukung jendela GUI; "
              f"gambar hasil tetap tersimpan sebagai PNG. Gunakan --no-gui untuk menghilangkan pesan ini.")
        return False

    tinggi, lebar = gray.shape
    dpi = 100
    lebar_fig = 3 * lebar / dpi
    skala = min(1.0, 24.0 / lebar_fig) if lebar_fig > 24.0 else 1.0
    fig, axs = plt.subplots(1, 3, figsize=(lebar_fig * skala, (tinggi / dpi + 0.5) * skala), dpi=dpi)

    axs[0].imshow(rgb, interpolation="nearest")
    axs[0].set_title("RGB (8 bit)")

    axs[1].imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axs[1].set_title("Grayscale (8 bit)")

    axs[2].imshow(biner, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axs[2].set_title("Biner")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    try:
        plt.show()
        return True
    except Exception as e:
        print(f"  [PERINGATAN] Jendela grafik tidak dapat ditampilkan: {e}")
        return False


# ---------------------------------------------------------------
# 5. Representasi dalam bentuk matriks
# ---------------------------------------------------------------
def tampilkan_matriks(rgb, gray, biner, batas=8):
    """Mencetak potongan matriks (batas x batas piksel pertama) ke console."""
    np.set_printoptions(linewidth=200)

    print("\n=== INFORMASI CITRA ===")
    print(f"Resolusi (lebar x tinggi) : {gray.shape[1]} x {gray.shape[0]} piksel")
    print(f"Matriks RGB       : {rgb.shape}  tipe {rgb.dtype}")
    print(f"Matriks Grayscale : {gray.shape}  tipe {gray.dtype}")
    print(f"Matriks Biner     : {biner.shape}  tipe {biner.dtype}")

    print(f"\n=== MATRIKS GRAYSCALE ({batas}x{batas} piksel pertama) ===")
    print(gray[:batas, :batas])

    print(f"\n=== MATRIKS BINER ({batas}x{batas} piksel pertama) ===")
    print(biner[:batas, :batas])

    for i, kanal in enumerate(["R", "G", "B"]):
        print(f"\n=== MATRIKS KANAL {kanal} ({batas}x{batas} piksel pertama) ===")
        print(rgb[:batas, :batas, i])


def simpan_matriks(rgb, gray, biner, folder=NAMA_FOLDER_OUTPUT, nama=NAMA_DASAR):
    """Menyimpan seluruh matriks (CSV) dan gambar hasil pemrosesan (PNG).

    Mengembalikan daftar dict: {nama, jenis, path, ukuran, status, error}.
    """
    os.makedirs(folder, exist_ok=True)
    daftar = []

    def simpan(nama_file, jenis, tulis):
        path_file = os.path.join(folder, nama_file)
        try:
            tulis(path_file)
            daftar.append({"nama": nama_file, "jenis": jenis, "path": path_file,
                           "ukuran": os.path.getsize(path_file), "status": "OK", "error": None})
        except Exception as e:
            daftar.append({"nama": nama_file, "jenis": jenis, "path": path_file,
                           "ukuran": None, "status": "GAGAL", "error": str(e)})

    # --- CSV: matriks penuh ---
    simpan(f"{nama}_grayscale.csv", "CSV matriks grayscale",
           lambda p: np.savetxt(p, gray, fmt="%d", delimiter=","))
    simpan(f"{nama}_biner.csv", "CSV matriks biner",
           lambda p: np.savetxt(p, biner, fmt="%d", delimiter=","))
    for i, kanal in enumerate("RGB"):
        simpan(f"{nama}_{kanal}.csv", f"CSV matriks kanal {kanal}",
               lambda p, i=i: np.savetxt(p, rgb[:, :, i], fmt="%d", delimiter=","))

    # --- PNG: gambar hasil pemrosesan ---
    simpan(f"{nama}_rgb.png", "PNG citra RGB",
           lambda p: Image.fromarray(rgb).save(p))
    for i, kanal in enumerate("RGB"):
        simpan(f"{nama}_{kanal}.png", f"PNG kanal {kanal}",
               lambda p, i=i: Image.fromarray(rgb[:, :, i]).save(p))
    simpan(f"{nama}_grayscale.png", "PNG citra grayscale",
           lambda p: Image.fromarray(gray).save(p))
    simpan(f"{nama}_biner.png", "PNG citra biner (0/255)",
           lambda p: Image.fromarray(biner.astype(np.uint8) * 255).save(p))

    for f in daftar:
        tanda = "OK    " if f["status"] == "OK" else "GAGAL "
        print(f"  [{tanda}] {f['nama']}  ({f['jenis']})")
    print(f"\nSemua matriks & gambar hasil disimpan di folder: {os.path.abspath(folder)}")
    return daftar


def verifikasi_output(folder, nama, gray, biner):
    """Membaca ulang PNG hasil simpan dan mencocokkannya dengan matriks.

    Menjamin gambar yang tersimpan benar-benar identik dengan matriks
    hasil pemrosesan (lossless). Mengembalikan dict {nama: True/False/'error: ..'}.
    """
    hasil = {}
    try:
        ulang = np.array(Image.open(os.path.join(folder, f"{nama}_grayscale.png")), dtype=np.uint8)
        hasil["grayscale"] = bool(np.array_equal(ulang, gray))
    except Exception as e:
        hasil["grayscale"] = f"error: {e}"
    try:
        ulang = np.array(Image.open(os.path.join(folder, f"{nama}_biner.png")), dtype=np.uint8)
        hasil["biner"] = bool(np.array_equal(ulang, biner.astype(np.uint8) * 255))
    except Exception as e:
        hasil["biner"] = f"error: {e}"
    return hasil


# ---------------------------------------------------------------
# 6. Laporan otomatis (Markdown)
# ---------------------------------------------------------------
def _fmt_ukuran(n):
    if n is None:
        return "-"
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _rb(n):
    """Format bilangan bulat dengan pemisah ribuan titik (gaya Indonesia)."""
    return f"{n:,}".replace(",", ".")


def buat_laporan(konteks):
    """Menyusun laporan Markdown dari konteks proses lalu menyimpannya.

    Mengembalikan path laporan, atau None bila gagal menulis.
    """
    status = konteks["status"]
    ikon = {"AMAN — SUKSES": "✅", "SELESAI DENGAN PERINGATAN": "⚠️", "GAGAL": "❌"}.get(status, "❌")
    info = konteks.get("info") or {}
    L = []
    tambah = L.append

    tambah("# Laporan Pemrosesan Citra Digital\n")
    tambah(f"- **Waktu proses** : {konteks['waktu']}")
    tambah(f"- **Program** : `program_pcd.py` versi {VERSI_PROGRAM}")
    tambah(f"- **Input** : `{konteks['path']}`")
    tambah(f"- **Durasi total** : {konteks['durasi_total']:.3f} detik")
    tambah(f"- **Status akhir** : {ikon} **{status}**\n")

    # --- 1. Informasi input ---
    tambah("## 1. Informasi Input\n")
    tambah("| Item | Nilai |")
    tambah("|---|---|")
    tambah(f"| Path | `{info.get('path', '-')}` |")
    tambah(f"| Format | {info.get('format', '-')} |")
    tambah(f"| Ukuran file | {_fmt_ukuran(info.get('ukuran_bytes'))} |")
    if info.get("lebar"):
        tambah(f"| Resolusi | {info['lebar']} x {info['tinggi']} piksel |")
        tambah(f"| Total piksel | {_rb(info['total_piksel'])} |")
    tambah("")

    # --- 2. Cara sistem berjalan ---
    tambah("## 2. Cara Sistem Berjalan (Pipeline)\n")
    tambah("| No | Tahap | Fungsi | Keterangan | Durasi | Status |")
    tambah("|---|---|---|---|---|---|")
    for i, t in enumerate(konteks["tahapan"], 1):
        ikon_t = "✅" if t["status"] == "OK" else "❌"
        catatan = f" — {t['catatan']}" if t.get("catatan") else ""
        tambah(f"| {i} | {t['nama']} | `{t['fungsi']}` | {t['ket']}{catatan} | "
               f"{t['durasi']:.3f} s | {ikon_t} {t['status']} |")
    tambah("")

    gray = konteks.get("gray")
    biner = konteks.get("biner")
    ambang = konteks.get("ambang")
    rgb = konteks.get("rgb")

    if gray is not None and biner is not None:
        putih = int(biner.sum())
        hitam = int(biner.size - putih)
        persen_putih = 100.0 * putih / biner.size if biner.size else 0.0
        tambah("### Penjelasan tiap tahap\n")
        tambah("1. **Membaca citra** — file didekode Pillow menjadi matriks RGB "
               f"`H×W×3` (uint8, 0–255) **tanpa resize**; resolusi keluaran = resolusi input.")
        tambah("2. **Grayscale** — tiap piksel diringkas jadi 1 kanal intensitas dengan rumus "
               "**Gray = 0.299·R + 0.587·G + 0.114·B** (bobot mengikuti kepekaan mata manusia, "
               "standar ITU-R BT.601).")
        tambah("3. **Ambang Otsu** — histogram 256 level dianalisis; dipilih nilai T yang "
               "memaksimalkan variansi antar-kelas gelap/terang. Hasil: **ambang = "
               f"{ambang}**.")
        tambah("4. **Biner** — piksel > ambang → 1 (putih), selainnya → 0 (hitam). "
               f"Hasil: **{_rb(putih)}** piksel putih ({persen_putih:.1f}%) dan "
               f"**{_rb(hitam)}** piksel hitam.")
        tambah("5. **Penyimpanan** — matriks lengkap diekspor ke CSV; gambar hasil diekspor "
               "ke PNG (lossless, matriks = piksel persis).")
        tambah("6. **Verifikasi** — PNG dibaca ulang lalu dicocokkan bit-per-bit dengan "
               "matriks di memori.\n")

    # --- 3. Statistik matriks ---
    if rgb is not None and gray is not None and biner is not None:
        putih = int(biner.sum())
        persen_putih = 100.0 * putih / biner.size if biner.size else 0.0
        tambah("## 3. Statistik Matriks\n")
        tambah("| Matriks | Bentuk | Tipe | Min | Maks | Rata-rata |")
        tambah("|---|---|---|---|---|---|")
        for i, kanal in enumerate("RGB"):
            m = rgb[:, :, i]
            tambah(f"| Kanal {kanal} | {m.shape[0]}×{m.shape[1]} | uint8 | {m.min()} | {m.max()} | {m.mean():.2f} |")
        tambah(f"| Grayscale | {gray.shape[0]}×{gray.shape[1]} | uint8 | {gray.min()} | {gray.max()} | {gray.mean():.2f} |")
        tambah(f"| Biner | {biner.shape[0]}×{biner.shape[1]} | uint8 | {biner.min()} | {biner.max()} | "
               f"{persen_putih:.2f}% putih |")
        tambah("")

    # --- 4. File output ---
    tambah("## 4. File Output yang Dihasilkan\n")
    tambah("| File | Jenis | Ukuran | Status |")
    tambah("|---|---|---|---|")
    for f in konteks.get("files", []):
        ikon_f = "✅" if f["status"] == "OK" else "❌"
        tambah(f"| `{f['nama']}` | {f['jenis']} | {_fmt_ukuran(f['ukuran'])} | {ikon_f} {f['status']} |")
    if not konteks.get("files"):
        tambah("| - | Tidak ada output (proses gagal lebih awal) | - | ❌ |")
    tambah("")

    # --- 5. Pemeriksaan keamanan / kesehatan sistem ---
    tambah("## 5. Pemeriksaan Keamanan & Kesehatan Sistem\n")
    pemeriksaan = susun_pemeriksaan(konteks)
    tambah("| No | Pemeriksaan | Hasil | Keterangan |")
    tambah("|---|---|---|---|")
    for i, (nama_p, ok, ket) in enumerate(pemeriksaan, 1):
        if ok is True:
            ikon_p, hasil_p = "✅", "LOLOS"
        elif ok is False:
            ikon_p, hasil_p = "❌", "GAGAL"
        else:
            ikon_p, hasil_p = "—", "TIDAK TERCAPAI"
        tambah(f"| {i} | {nama_p} | {ikon_p} {hasil_p} | {ket} |")
    tambah("")

    # --- 6. Peringatan ---
    tambah("## 6. Peringatan / Catatan\n")
    if konteks.get("peringatan"):
        for w in konteks["peringatan"]:
            tambah(f"- ⚠️ {w}")
    else:
        tambah("- Tidak ada peringatan.")
    if konteks.get("error"):
        tambah(f"- ❌ **Error:** {konteks['error']}")
    tambah("")

    # --- 7. Kesimpulan ---
    tambah("## 7. Kesimpulan\n")
    if status == "AMAN — SUKSES":
        tambah("Sistem berjalan **AMAN**. Seluruh tahap pipeline (baca → grayscale → Otsu → "
               "biner → simpan → verifikasi) selesai tanpa error, semua output tersimpan, dan "
               "gambar hasil terverifikasi identik dengan matriksnya.")
    elif status == "SELESAI DENGAN PERINGATAN":
        tambah("Sistem berjalan **AMAN dengan catatan**: pemrosesan selesai dan output tersimpan, "
               "tetapi ada peringatan pada bagian 6 yang perlu diperhatikan.")
    else:
        tambah("Sistem **TIDAK berjalan dengan aman** (status GAGAL). Periksa bagian 2 dan 6 "
               "untuk mengetahui tahap yang gagal beserta penyebabnya, lalu perbaiki dan jalankan ulang.")
    tambah("")

    # --- Lampiran: potongan matriks ---
    if gray is not None:
        tambah("## Lampiran: Potongan Matriks (8×8 piksel pertama)\n")
        tambah("```")
        tambah("Grayscale:")
        tambah(np.array2string(gray[:8, :8], separator=", ", max_line_width=100))
        tambah("")
        tambah("Biner:")
        tambah(np.array2string(biner[:8, :8], separator=", ", max_line_width=100))
        tambah("```\n")

    # --- Tulis file ---
    try:
        path_laporan = os.path.join(konteks["folder_output"], NAMA_LAPORAN)
        with open(path_laporan, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        print(f"\nLaporan disimpan: {os.path.abspath(path_laporan)}")
        return path_laporan
    except Exception as e:
        print(f"[PERINGATAN] Gagal menyimpan laporan: {e}")
        return None


def susun_pemeriksaan(konteks):
    """Menyusun daftar (nama, True/False/None, keterangan) untuk tabel pemeriksaan."""
    info = konteks.get("info") or {}
    tahapan = konteks["tahapan"]
    files = konteks.get("files") or []
    verif = konteks.get("verifikasi") or {}
    gray = konteks.get("gray")
    biner = konteks.get("biner")
    ambang = konteks.get("ambang")

    def tahap_ok(nama):
        for t in tahapan:
            if t["nama"] == nama:
                return t["status"] == "OK"
        return None

    p = []
    p.append(("Library inti terpasang (numpy + Pillow)", True,
              f"numpy {np.__version__}, Pillow {Image.__version__}"))
    p.append(("File input ditemukan", info.get("ukuran_bytes") is not None,
              info.get("path") or "tidak tercapai"))
    p.append(("Format file didukung (bmp/jpg/png)", info.get("format") is not None,
              info.get("format") or "tidak tercapai"))
    if info.get("ukuran_bytes") is not None:
        p.append((f"Ukuran file wajar (≤ {BATAS_UKURAN_FILE_MB} MB)",
                  info.get("ukuran_ok"), _fmt_ukuran(info.get("ukuran_bytes"))))
    if info.get("lebar"):
        p.append((f"Resolusi wajar (≤ {BATAS_SISI_PIKSEL} px sisi, ≤ {_rb(BATAS_TOTAL_PIKSEL)} px total)",
                  info.get("resolusi_ok"), f"{info['lebar']}×{info['tinggi']}"))
    p.append(("Citra berhasil didekode tanpa error", tahap_ok("Membaca citra"), "-"))
    if gray is not None:
        gray_ok = gray.dtype == np.uint8 and gray.ndim == 2
        p.append(("Grayscale valid (uint8, satu kanal 0–255)", gray_ok,
                  f"bentuk {gray.shape}, dtype {gray.dtype}"))
    if ambang is not None:
        ambang_ok = 0 <= int(ambang) <= 255
        p.append(("Ambang Otsu valid (0–255)", ambang_ok, f"ambang = {ambang}"))
    if biner is not None:
        nilai_unik = set(np.unique(biner).tolist())
        p.append(("Matriks biner valid (hanya 0/1)", nilai_unik.issubset({0, 1}),
                  f"nilai unik: {sorted(nilai_unik)}"))
    csv_files = [f for f in files if f["nama"].endswith(".csv")]
    png_files = [f for f in files if f["nama"].endswith(".png")]
    p.append(("Semua CSV matriks tersimpan (5 file)",
              all(f["status"] == "OK" for f in csv_files) and len(csv_files) == 5,
              f"{sum(f['status'] == 'OK' for f in csv_files)}/{len(csv_files) or 0} berhasil"))
    p.append(("Semua PNG gambar hasil tersimpan (6 file)",
              all(f["status"] == "OK" for f in png_files) and len(png_files) == 6,
              f"{sum(f['status'] == 'OK' for f in png_files)}/{len(png_files) or 0} berhasil"))
    if verif:
        gs = verif.get("grayscale")
        bn = verif.get("biner")
        p.append(("PNG terverifikasi identik dengan matriks (readback)",
                  gs is True and bn is True,
                  f"grayscale: {gs}, biner: {bn}"))
    p.append(("Tidak ada error runtime", konteks.get("error") is None,
              konteks.get("error") or "bersih"))
    return p


# ---------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------
def proses(path, tampilkan_gui=True, folder=NAMA_FOLDER_OUTPUT):
    """Menjalankan seluruh pipeline dan mengembalikan dict konteks laporan."""
    konteks = {
        "waktu": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "path": path, "status": "GAGAL", "error": None, "durasi_total": 0.0,
        "info": {}, "ambang": None, "rgb": None, "gray": None, "biner": None,
        "tahapan": [], "peringatan": [], "files": [], "verifikasi": {},
        "folder_output": folder, "laporan_path": None, "gui": None,
    }
    awal = time.perf_counter()

    def jalankan(nama, fungsi, ket, fn):
        t0 = time.perf_counter()
        try:
            hasil = fn()
        except Exception as e:
            konteks["tahapan"].append({"nama": nama, "fungsi": fungsi, "ket": ket,
                                       "status": "GAGAL", "durasi": time.perf_counter() - t0,
                                       "catatan": str(e)})
            raise
        konteks["tahapan"].append({"nama": nama, "fungsi": fungsi, "ket": ket,
                                   "status": "OK", "durasi": time.perf_counter() - t0,
                                   "catatan": None})
        return hasil

    error = None
    try:
        # --- Validasi & info file ---
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        ekstensi = os.path.splitext(path)[1].lower()
        if ekstensi not in FORMAT_DIDUKUNG:
            raise ValueError(f"Format '{ekstensi}' tidak didukung. Gunakan: bmp, jpg, png.")
        ukuran = os.path.getsize(path)
        ukuran_ok = ukuran <= BATAS_UKURAN_FILE_MB * 1024 * 1024
        if not ukuran_ok:
            konteks["peringatan"].append(
                f"Ukuran file {_fmt_ukuran(ukuran)} melebihi {BATAS_UKURAN_FILE_MB} MB — "
                "pemrosesan dan penulisan CSV bisa lambat/memakan memori besar.")
        konteks["info"].update({"path": os.path.abspath(path), "format": ekstensi,
                                "ukuran_bytes": ukuran, "ukuran_ok": ukuran_ok})

        # --- Pipeline ---
        rgb = jalankan("Membaca citra", "baca_citra()",
                       "Dekode file menjadi matriks RGB H×W×3 (uint8), tanpa resize",
                       lambda: baca_citra(path))
        konteks["rgb"] = rgb
        tinggi, lebar = rgb.shape[0], rgb.shape[1]
        total_piksel = tinggi * lebar
        resolusi_ok = max(tinggi, lebar) <= BATAS_SISI_PIKSEL and total_piksel <= BATAS_TOTAL_PIKSEL
        if not resolusi_ok:
            konteks["peringatan"].append(
                f"Resolusi {lebar}×{tinggi} ({_rb(total_piksel)} piksel) sangat besar — "
                "ekspor CSV dapat memakan waktu lama.")
        konteks["info"].update({"lebar": lebar, "tinggi": tinggi,
                                "total_piksel": total_piksel, "resolusi_ok": resolusi_ok})
        print(f"\nCitra terbaca: {lebar} x {tinggi} piksel, "
              f"{_fmt_ukuran(ukuran)}, format {ekstensi}")

        gray = jalankan("Konversi grayscale", "ke_grayscale()",
                        "Gray = 0.299R + 0.587G + 0.114B (BT.601), 3 kanal -> 1 kanal",
                        lambda: ke_grayscale(rgb))
        konteks["gray"] = gray

        ambang = jalankan("Hitung ambang Otsu", "hitung_otsu()",
                          "Cari T yang memaksimalkan variansi antar-kelas histogram",
                          lambda: hitung_otsu(gray))
        konteks["ambang"] = ambang
        print(f"Nilai ambang (Otsu): {ambang}")
        if ambang in (0, 255):
            konteks["peringatan"].append(
                f"Ambang Otsu ekstrem ({ambang}) — citra kemungkinan nyaris satu warna; "
                "hasil biner mungkin satu warna semua.")

        biner = jalankan("Konversi biner", "ke_biner()",
                         f"Piksel > {ambang} -> 1 (putih), selainnya -> 0 (hitam)",
                         lambda: ke_biner(gray, ambang)[0])
        konteks["biner"] = biner

        tampilkan_matriks(rgb, gray, biner)

        files = jalankan("Simpan CSV & PNG", "simpan_matriks()",
                         "Ekspor 5 CSV matriks penuh + 6 PNG gambar hasil",
                         lambda: simpan_matriks(rgb, gray, biner, folder=folder))
        konteks["files"] = files
        gagal_simpan = [f["nama"] for f in files if f["status"] != "OK"]
        if gagal_simpan:
            konteks["peringatan"].append(f"{len(gagal_simpan)} file output gagal disimpan: "
                                         ", ".join(gagal_simpan))

        verif = jalankan("Verifikasi output", "verifikasi_output()",
                         "Baca ulang PNG, cocokkan bit-per-bit dengan matriks",
                         lambda: verifikasi_output(folder, NAMA_DASAR, gray, biner))
        konteks["verifikasi"] = verif
        for k, v in verif.items():
            if v is not True:
                konteks["peringatan"].append(f"Verifikasi readback {k} tidak lolos: {v}")

        # --- Grafik (opsional) ---
        if tampilkan_gui:
            gui_ok = jalankan("Tampilkan grafik", "tampilkan()",
                              "Jendela matplotlib RGB | Grayscale | Biner",
                              lambda: tampilkan(rgb, gray, biner))
            konteks["gui"] = gui_ok
            if gui_ok is False:
                konteks["peringatan"].append(
                    "Jendela grafik tidak dapat ditampilkan di lingkungan ini "
                    "(output gambar tetap tersimpan sebagai PNG).")

        # --- Status akhir ---
        if konteks["peringatan"]:
            konteks["status"] = "SELESAI DENGAN PERINGATAN"
        else:
            konteks["status"] = "AMAN — SUKSES"

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        konteks["error"] = error
        konteks["status"] = "GAGAL"
        print(f"\n[ERROR] {error}")

    konteks["durasi_total"] = time.perf_counter() - awal
    konteks["laporan_path"] = buat_laporan(konteks)

    # --- Ringkasan console ---
    print("\n" + "=" * 60)
    print(f"STATUS AKHIR : {konteks['status']}")
    if konteks["peringatan"]:
        for w in konteks["peringatan"]:
            print(f"  PERINGATAN : {w}")
    if konteks["laporan_path"]:
        print(f"LAPORAN      : {os.path.abspath(konteks['laporan_path'])}")
    print("=" * 60)
    return konteks


# ---------------------------------------------------------------
# Program utama
# ---------------------------------------------------------------
def parse_argumen(argv):
    p = argparse.ArgumentParser(
        description="Pengolahan citra digital: RGB -> grayscale -> biner (Otsu) "
                    "+ laporan otomatis.")
    p.add_argument("path", nargs="?", default=None,
                   help="Path gambar (bmp/jpg/png). Kosongkan untuk mode input().")
    p.add_argument("--no-gui", action="store_true",
                   help="Jangan tampilkan jendela matplotlib (hanya simpan file).")
    p.add_argument("--folder", default=NAMA_FOLDER_OUTPUT,
                   help=f"Folder output (default: {NAMA_FOLDER_OUTPUT}).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_argumen(sys.argv[1:])
    _path = args.path
    if not _path:
        try:
            _path = input("Masukkan path gambar (bmp/jpg/png): ").strip().strip('"').strip("'")
        except EOFError:
            print("Input tidak tersedia. Jalankan: python program_pcd.py <path_gambar>")
            sys.exit(1)
    if not _path:
        print("Path kosong. Dibatalkan.")
        sys.exit(1)

    _konteks = proses(_path, tampilkan_gui=not args.no_gui, folder=args.folder)
    sys.exit(0 if _konteks["status"] != "GAGAL" else 1)
