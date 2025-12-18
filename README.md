# Dice Coefficient Segmentation Evaluator (Ground Truth vs Prediksi)

Program ini menghitung **Dice Coefficient** untuk mengevaluasi kualitas hasil segmentasi citra (mask) dengan membandingkan **Ground Truth** (label manual) dan **Prediksi** (hasil algoritma). Selain skor, program juga menampilkan **visualisasi error** berwarna:

- **Hijau** = benar terdeteksi (**True Positive / Intersection**)
- **Merah** = salah prediksi / noise (**False Positive**)
- **Biru** = objek terlewat (**False Negative**)

---

## 1) Fitur Utama

- Baca dua citra mask (grayscale) dari path.
- Binarisasi otomatis dengan threshold **127** (pixel > 127 dianggap objek).
- Jika ukuran citra berbeda, prediksi akan **di-resize** mengikuti Ground Truth.
- Hitung **Dice Coefficient**:
  \[
  Dice = \frac{2 \cdot |GT \cap Pred|}{|GT| + |Pred|}
  \]
- Tampilkan 3 panel visual:
  1. Ground Truth
  2. Prediksi
  3. Visualisasi error + skor Dice

---

## 2) Kebutuhan (Dependencies)

Pastikan environment punya:

- Python 3.x
- numpy
- opencv-python
- matplotlib

Instalasi (lokal):
```bash
pip install numpy opencv-python matplotlib
```

MIT Licence
