#Aplikasi Regresi untuk Pemecahan Problem
#Metode Model Pangkat Sederhana

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

#Data sampel yang diambil dari dataset
durasi_waktu_belajar = np.array([7, 4, 8, 5, 7, 3, 7, 8, 5, 4])
nilai_ujian = np.array([91.0, 65.0, 45.0, 36.0, 66.0, 61.0, 63.0, 42.0, 61.0, 69.0])

#Membuat model Regresi Pangkat Sederhana
def regresi_pangkat_sederhana(x, y):
    x_log = np.log(x)
    y_log = np.log(y)
    koefisien = np.polyfit(x_log, y_log, 1)
    return koefisien

#Melakukan regresi
koefisien = regresi_pangkat_sederhana(durasi_waktu_belajar, nilai_ujian)
a = np.exp(koefisien[1])
b = koefisien[0]

#Menghitung nilai prediksi
y_pred = a * (durasi_waktu_belajar ** b)

#Menghitung galat RMS
galat_rms = np.sqrt(mean_squared_error(nilai_ujian, y_pred))

#Menampilkan koefisien regresi, intercept, dan galat RMS
print("Koefisien Regresi:", np.round(a, 4))
print("Intercept:", np.round(b, 4))
print("Galat RMS untuk Model Pangkat Sederhana:", galat_rms)

#Menyusun hasil regresi dan perhitungan galat RMS ke dalam DataFrame
df_result = pd.DataFrame({
    'Durasi Waktu Belajar': durasi_waktu_belajar,
    'Nilai Ujian Asli': nilai_ujian,
    'Nilai Ujian Prediksi': y_pred
})

#Menampilkan hasil regresi dan perhitungan galat RMS dalam bentuk DataFrame
print(df_result.to_string(index=False))

#Plot titik data dan hasil regresi
plt.scatter(durasi_waktu_belajar, nilai_ujian, color='black', label='Data Asli')
plt.plot(np.sort(durasi_waktu_belajar), y_pred[np.argsort(durasi_waktu_belajar)], color='purple', label='Regresi')
plt.xlabel('Durasi Waktu Belajar')
plt.ylabel('Nilai Ujian')
plt.title(f'Regresi Pangkat Sederhana (Galat RMS: {galat_rms:.5f})')
plt.legend()
plt.grid(True)
plt.show()