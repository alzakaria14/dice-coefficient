import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def calculate_dice_score(ground_truth, prediction):
    """
    Menghitung Dice Coefficient.
    Rumus: 2 * (Irisan) / (Total Pixel GT + Total Pixel Prediksi)
    """
    # 1. Normalisasi ke biner (0 dan 1)
    # Anggap pixel > 127 adalah objek (1), sisanya background (0)
    gt = (ground_truth > 127).astype(np.float32)
    pred = (prediction > 127).astype(np.float32)

    # 2. Hitung Irisan (Intersection)
    # Perkalian elemen: 1 x 1 = 1 (Irisan), sisanya 0
    intersection = np.sum(gt * pred)

    # 3. Hitung Total Area
    total_area = np.sum(gt) + np.sum(pred)

    # 4. Hitung Dice (tambah epsilon kecil agar tidak error bagi 0)
    epsilon = 1e-6
    dice = (2. * intersection + epsilon) / (total_area + epsilon)

    return dice

def visualize_difference(gt, pred):
    """
    Membuat visualisasi perbedaan:
    - Hijau: Cocok (True Positive)
    - Merah: Salah Prediksi (False Positive / Noise)
    - Biru: Terlewat (False Negative / Objek tidak terdeteksi)
    """
    # Pastikan ukuran sama
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # Binarisasi
    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)

    # Buat citra RGB kosong
    diff_img = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    # Logika Pewarnaan
    # Area Irisan (Benar) -> Hijau
    intersection = cv2.bitwise_and(gt_bin, pred_bin)

    # Area Salah Prediksi (Ada di Prediksi, Tidak ada di GT) -> Merah
    false_positive = cv2.bitwise_and(pred_bin, cv2.bitwise_not(gt_bin))

    # Area Terlewat (Ada di GT, Tidak ada di Prediksi) -> Biru
    false_negative = cv2.bitwise_and(gt_bin, cv2.bitwise_not(pred_bin))

    # Masukkan ke channel warna (BGR di OpenCV, tapi nanti di plot jadi RGB)
    # Channel 0: Merah (di Matplotlib) -> Kita isi False Positive
    diff_img[:, :, 0] = false_positive
    # Channel 1: Hijau (di Matplotlib) -> Kita isi Intersection
    diff_img[:, :, 1] = intersection
    # Channel 2: Biru (di Matplotlib) -> Kita isi False Negative
    diff_img[:, :, 2] = false_negative

    return diff_img

# --- MAIN PROGRAM ---

print("=== PROGRAM MENGHITUNG DICE COEFFICIENT ===")
print("Langkah 1: Masukkan path Citra Ground Truth (Kunci Jawaban Manual)")
path_gt = input("Path Ground Truth: ").strip()

print("\nLangkah 2: Masukkan path Citra Hasil Segmentasi (Prediksi Algoritma)")
path_pred = input("Path Prediksi: ").strip()

# Proses file yang diupload
if os.path.exists(path_gt) and os.path.exists(path_pred):
    # Baca gambar sebagai Grayscale
    img_gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    img_pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
    
    if img_gt is None or img_pred is None:
        print("Error: Gagal membaca file gambar. Periksa path file.")
    else:

    # Validasi Ukuran
    # Jika ukuran beda, samakan ukuran prediksi ke ground truth
        if img_gt.shape != img_pred.shape:
            print(f"\n[INFO] Ukuran berbeda. Resize prediksi {img_pred.shape} ke {img_gt.shape}...")
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))

        # Hitung Skor
        score = calculate_dice_score(img_gt, img_pred)

        # Buat Visualisasi Error
        vis_diff = visualize_difference(img_gt, img_pred)

        # Tampilkan Hasil
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Ground Truth (Manual)")
        plt.imshow(img_gt, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Prediksi (Algoritma)")
        plt.imshow(img_pred, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Visualisasi Error\nDice Score: {score:.4f}")
        plt.imshow(vis_diff)
        plt.xlabel("Hijau: Benar | Merah: Noise | Biru: Terlewat")
        # Hilangkan ticks tapi biarkan label
        plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()

        print(f"\n--- HASIL AKHIR ---")
        print(f"Dice Coefficient: {score:.5f}")
        if score > 0.7:
            print("Kualitas Segmentasi: BAIK")
        else:
            print("Kualitas Segmentasi: BURUK (Perlu perbaikan threshold/metode)")

else:
    print("Error: File tidak ditemukan. Periksa path file.")