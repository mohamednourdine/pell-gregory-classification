import pandas as pd
import math

def eğim_hesapla(x1, y1, x2, y2):
    # İki nokta arasındaki eğimi hesapla
    eğim = (y2 - y1) / (x2 - x1)
    return eğim

# Excel dosyasından veriyi oku
veri = pd.read_excel("C:\\Users\\MasterChef\\Desktop\\data_37_38.xlsx")

# Sonuçları tutacak bir liste oluştur
sonuçlar = []

# Veri setindeki her satır için işlem yap
for index, row in veri.iterrows():
    # İki noktanın koordinatlarını seç
    x1, y1 = row['Column1'], row['Column2']
    x2, y2 = row['Column3'], row['Column4']

    # Sonraki iki noktanın koordinatlarını seç
    x3, y3 = row['Column5'], row['Column6']
    x4, y4 = row['Column7'], row['Column8']

    # Eğimleri hesapla
    hesaplanan_eğim1 = eğim_hesapla(x1, y1, x2, y2)
    hesaplanan_eğim2 = eğim_hesapla(x3, y3, x4, y4)

    # Açıları hesapla (radyan cinsinden)
    theta1 = math.atan(hesaplanan_eğim1)
    theta2 = math.atan(hesaplanan_eğim2)

    # İki açı arasındaki farkı alarak tan(x) değerini hesapla
    tan_x_değeri = math.tan(theta1 - theta2)

    # Arctan(x) değerini hesapla (radyan cinsinden)
    arctan_x_değeri = math.atan(tan_x_değeri)

    # Arctan(x) değerini dereceye çevir
    arctan_x_derece = math.degrees(arctan_x_değeri)

    # Sonuçları liste olarak ekleyin
    sonuçlar.append({
        'Eğim 1': hesaplanan_eğim1,
        'Eğim 2': hesaplanan_eğim2,
        'Arctan(x) Değeri(radyan)': arctan_x_değeri,
        'Arctan(x) Değeri (Derece)': arctan_x_derece
    })

# Liste üzerinden bir DataFrame oluşturun
sonuçlar_df = pd.DataFrame(sonuçlar)

# Sonuçları Excel dosyasına yaz
sonuçlar_df.to_excel("C:\\Users\\MasterChef\\Desktop\\sonuclar.xlsx", index=False)
print("Sonuçlar başarıyla kaydedildi.")
