# Helipad Detection & Training Project 🚁🎯

Repositori ini berisi skrip pelatihan (*training script*), konfigurasi, dan *dataset* untuk melatih model deteksi objek berbasis **YOLOv8** guna mengenali area pendaratan helipad secara presisi untuk kebutuhan navigasi dan pendaratan otonom drone.

---

## 🚀 Fitur Utama
* **Training YOLOv8:** Skrip pelatihan kustom untuk mendeteksi helipad menggunakan arsitektur YOLOv8.
* **Dataset Terstruktur:** Berisi direktori *train*, *val*, dan *test* lengkap dengan label bounding box yang siap digunakan untuk eksperimen *computer vision*.
* **Bobot Model & Hasil Evaluasi:** Menyimpan hasil metrik evaluasi dan bobot model hasil pelatihan.

---

## 🛠️ Struktur Direktori Proyek
```text
Helli pad/
├── config/             # Konfigurasi model dan dataset (YAML)
├── core_ai/            # Skrip inti AI, pelatihan, dan deteksi
├── datasets/           # Dataset gambar dan label (train, val, test)
├── runs/               # Hasil *training* dan kurva metrik evaluasi
├── logs/               # Log sistem dan sampel augmentasi
└── requirements.txt    # Daftar dependensi pustaka Python
```

---

## ⚙️ Cara Menjalankan Pelatihan (Training)

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/MRZufarrel/Helipad.git
   cd Helipad
   ```

2. **Install dependensi yang diperlukan:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan skrip pelatihan YOLOv8:**
   ```bash
   python core_ai/train.py
   ```

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah **MIT License**. Silakan lihat file [LICENSE](LICENSE) untuk informasi lebih lanjut.
