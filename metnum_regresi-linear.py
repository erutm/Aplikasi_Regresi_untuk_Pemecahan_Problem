#Aplikasi Regresi untuk Pemecahan Problem
#Metode Model Linear

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Data sampel yang diambil dari dataset
data = {
    'Durasi Waktu Belajar': [7, 4, 8, 5, 7, 3, 7, 8, 5, 4],
    'Nilai Ujian': [91.0, 65.0, 45.0, 36.0, 66.0, 61.0, 63.0, 42.0, 61.0, 69.0]
}

#Membuat DataFrame
df = pd.DataFrame(data)

#Fungsi untuk mengimplementasikan regresi linear, plot hasil, dan perhitungan galat RMS
def regresi_linear(df, x_column, y_column):
    X = df[x_column].values.reshape(-1, 1)
    y = df[y_column].values
    
    #Membuat model regresi linear
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    #Menghitung galat RMS
    galat_rms = np.sqrt(mean_squared_error(y, y_pred))
    
    #Plot data dan garis regresi
    plt.scatter(X, y, color='green', label='Data')
    plt.plot(X, y_pred, color='black', label='Regresi Linear')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Regresi Linear (Galat RMS: {galat_rms:.5f})')
    plt.legend()
    plt.show()
    
    # Menampilkan hasil regresi dan galat RMS
    print("Koefisien Regresi: ", model.coef_[0])
    print("Intercept: ", model.intercept_)
    print(f'Galat RMS untuk Model Linear: {galat_rms}')
    
    return model, y_pred, galat_rms

# Menjalankan fungsi regresi linear dan plot hasilnya
model, y_pred, galat_rms = regresi_linear(df, 'Durasi Waktu Belajar', 'Nilai Ujian')

# Menampilkan hasil prediksi
prediksi_df = pd.DataFrame({'Durasi Waktu Belajar': df['Durasi Waktu Belajar'], 'Nilai Ujian Asli': df['Nilai Ujian'], 'Nilai Ujian Prediksi': y_pred})
print(prediksi_df.to_string(index=False))