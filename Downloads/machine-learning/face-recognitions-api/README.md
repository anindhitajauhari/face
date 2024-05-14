# Face Recognition

Change directory (cd) ke folder face-recognition

```bash
cd face-recognition
```

## 1. Mengumpulkan Dataset dengan Kamera

```bash
python facerec.py -d nama_dataset -s 0
```
Setelah muncul window, tekan spacebar untuk mulai mengumpulkan dataset. Jika sudah selesai, tekan tombol ESC pada keyboard.

## 2. Training Data dengan Algoritma KNN

```bash
python retrain.py t -m trained1_model.clf -s 0
```

## 3. Run Face Recognition dengan Webcam

```bash
python retrain.py r -m trained1_model.clf -s 0
```

Ganti bagian ```-s``` 0 dengan ```-s "path video"``` untuk menggunakan video sebagai sourcenya (bukan kamera).
