# Sistem Pembayaran Sekolah Anugerah Insani

Aplikasi lokal berbasis Streamlit + SQLite dengan analitik dan prediksi risiko keterlambatan pembayaran.

## Menjalankan
1. Instal Python 3.10 atau lebih baru.
2. Ekstrak ZIP dan buka terminal di folder ini.
3. (Opsional) Buat virtual environment: `python -m venv .venv` dan aktifkan.
4. Instal paket: `pip install -r requirements.txt`
5. Jalankan: `streamlit run app.py`
6. Buka alamat yang muncul (umumnya http://localhost:8501).

Database `sekolah.db` dibuat otomatis. Buka menu **Data Simulasi** untuk membuat contoh data.

## Fitur
- CRUD siswa; tagihan bulanan dan jenis biaya lain; pencatatan cicilan dengan validasi batas tagihan.
- Dashboard, analisis kelas dan tren penerimaan, ekspor CSV, bukti transaksi yang dapat dicetak dari browser.
- Machine Learning RandomForest untuk prediksi risiko telat bayar dari riwayat *tagihan yang sudah jatuh tempo*; evaluasi holdout jika data cukup. Jika data tidak cukup, aplikasi menampilkan status belum dapat dilatih, bukan prediksi palsu.

## Penting
- Data demo adalah **simulasi**, bukan informasi sekolah sesungguhnya.
- Model prediksi adalah indikator pendukung keputusan, bukan penentu sanksi atau kelayakan siswa.
- Ini prototipe untuk pemakaian lokal oleh satu operator tepercaya, **belum memiliki autentikasi, kontrol akses, enkripsi, atau mekanisme backup otomatis**. Jangan publikasikan ke internet atau unggah data siswa nyata ke hosting publik tanpa pengamanan, kebijakan privasi, dan audit.
- Untuk produksi: tambah login berbasis peran, TLS, backup terenkripsi, audit trail, pengujian, dan pemisahan database/server.

## Random Forest dengan dataset 500 siswa (versi eksperimen)
1. Impor file `Dataset_Monitoring_Pembayaran_500_Siswa.xlsx` melalui menu Upload Dataset Excel.
2. Buka menu Prediksi Machine Learning.
3. Klik **Jalankan Random Forest — data simulasi historis**.
4. Aplikasi membuat 6 bulan riwayat sintetis per siswa **di memori**, tanpa mengubah database, lalu menampilkan evaluasi model, tabel skor simulasi, dan ekspor CSV.
5. Model data asli tetap dinonaktifkan sampai ada contoh riwayat terlambat yang valid. Jangan menggunakan skor simulasi untuk keputusan terhadap siswa.

Catatan: semua tagihan pada Excel asli jatuh tempo 10 November 2026; pembayaran tercatat pada 1–2 November 2026. Karena itu tidak tersedia contoh terlambat asli. Skor simulasi adalah demonstrasi teknis, bukan prediksi valid untuk siswa nyata.
