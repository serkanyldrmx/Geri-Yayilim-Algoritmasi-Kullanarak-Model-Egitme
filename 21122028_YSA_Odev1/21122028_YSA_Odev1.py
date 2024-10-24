import numpy as np
import matplotlib.pyplot as plt

# Aktivasyon fonksiyonları
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_turevi(x):
    return x * (1 - x)

# Egitim verileri
girdi = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

hedef = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
#sabitler
ogrenme_hizi = 0.1
donem = 10000  

gizli_noron_sayisi = int(input("Gizli katmandaki nöron sayisini giriniz: "))

# Agirliklari rastgele baslatıldı
np.random.seed(1)
agirliklar_girdi_gizli = np.random.rand(girdi.shape[1], gizli_noron_sayisi)
agirliklar_gizli_cikis = np.random.rand(gizli_noron_sayisi, hedef.shape[1])

# Geri yayilim algoritmasi
for donem in range(donem):
    gizli_katman_girdisi = np.dot(girdi, agirliklar_girdi_gizli)
    gizli_katman_cikisi = sigmoid(gizli_katman_girdisi)
    
    cikis_katmani_girdisi = np.dot(gizli_katman_cikisi, agirliklar_gizli_cikis)
    cikis = sigmoid(cikis_katmani_girdisi)
    
    hata = hedef - cikis
    
    if donem % 1000 == 0:
        toplam_hata = np.sum(hata**2)
        print(f"Donem {donem}, Toplam Hata: {toplam_hata}")
    
    # Geri yayilim
    d_cikis = hata * sigmoid_turevi(cikis)
    hata_gizli_katman = d_cikis.dot(agirliklar_gizli_cikis.T)
    d_gizli_katman = hata_gizli_katman * sigmoid_turevi(gizli_katman_cikisi)
    
    # Agirlik güncellemeleri
    agirliklar_gizli_cikis += gizli_katman_cikisi.T.dot(d_cikis) * ogrenme_hizi
    agirliklar_girdi_gizli += girdi.T.dot(d_gizli_katman) * ogrenme_hizi

toplam_hata = np.sum(hata**2)
print(f"Egitim Tamamlandi, Toplam Hata: {toplam_hata}")

# Ağ yapısı görseli
def plot_network(input_size, hidden_size, output_size):
    plt.figure(figsize=(12, 8))
    
    for i in range(input_size):
        plt.scatter(1, i + 1, s=1000, c='lightblue', edgecolor='black')
        plt.text(1, i + 1, f"Girdi {i+1}", ha='center', va='center', fontsize=12)
    
    for i in range(hidden_size):
        plt.scatter(2, i + 1, s=1000, c='lightgreen', edgecolor='black')
        plt.text(2, i + 1, f"Gizli {i+1}", ha='center', va='center', fontsize=12)
        
    for i in range(output_size):
        plt.scatter(3, i + 1, s=1000, c='salmon', edgecolor='black')
        plt.text(3, i + 1, f"Çıkış {i+1}", ha='center', va='center', fontsize=12)

    for i in range(input_size):
        for j in range(hidden_size):
            plt.plot([1, 2], [i + 1, j + 1], c='gray', linestyle='dashed')
            
    for i in range(hidden_size):
        for j in range(output_size):
            plt.plot([2, 3], [i + 1, j + 1], c='gray', linestyle='dashed')

    plt.xlim(0.5, 3.5)
    plt.ylim(0.5, max(input_size, hidden_size, output_size) + 0.5)
    plt.axis('off')
    plt.title('Yapay Sinir Ağı Yapısı', fontsize=16)
    plt.show()

# Ağ yapısı
plot_network(girdi.shape[1], gizli_noron_sayisi, hedef.shape[1])
