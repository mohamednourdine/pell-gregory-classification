import pandas as pd
from sympy import symbols, Eq, solve

# Bul fonksiyonunun tanımlandığı varsayılan bir örnek
def bul(x1, y1, x2, y2, x5):
    m = (y2 - y1) / (x2 - x1)
    
    x, y = symbols('x y')
    eq1 = Eq(y - y2 - (m * x - m * x2), 0)
    eq2 = Eq(x - x5, 0)
    
    solution = solve([eq1, eq2], [x, y])
    
    x_value = solution[x]
    y_value = solution[y]
    
    y_value = round(y_value)
    
    return y_value

# Excel dosyasını oku
filename = 'C:/Users/MasterChef/Desktop/data_47_48.xlsx'
data_47_48 = pd.read_excel(filename)

# Y5_47_48_veri.xlsx dosyasını oku
filename1 = 'C:/Users/MasterChef/Desktop/Y5_47_48_veri.xlsx'
Y5_47_48 = pd.read_excel(filename1)

# Boyutu al
kk = data_47_48.shape[0]

# Veriyi işle
for i in range(kk):
    x1 = data_47_48.loc[i, 'Column1']
    y1 = data_47_48.loc[i, 'Column2']
    x2 = data_47_48.loc[i, 'Column3']
    y2 = data_47_48.loc[i, 'Column4']
    x5 = data_47_48.loc[i, 'Column9']
    
    y = bul(x1, y1, x2, y2, x5)
    
    Y5_47_48.loc[i, 'Column2'] = y
    
    x1 = data_47_48.loc[i, 'Column5']
    y1 = data_47_48.loc[i, 'Column6']
    x2 = data_47_48.loc[i, 'Column7']
    y2 = data_47_48.loc[i, 'Column8']
    x5 = data_47_48.loc[i, 'Column9']
    
    y = bul(x1, y1, x2, y2, x5)
    
    Y5_47_48.loc[i, 'Column3'] = y

# İşlenmiş veriyi yaz
Y5_47_48.to_excel('C:/Users/MasterChef/Desktop/Y5_47_48_sonuc.xlsx', index=False)
