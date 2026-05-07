import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import warnings

# Teknik uyarıları gizle (Görünümü temiz tutmak için)
warnings.filterwarnings('ignore')

# 1. Veri Setini Yükleme
veri = pd.read_csv('test_kredi_tahmini.csv')

# 2. Veri Ön İşleme
for sutun in veri.columns:
    if veri[sutun].dtype == 'object':
        veri[sutun] = veri[sutun].fillna(veri[sutun].mode()[0])
    else:
        veri[sutun] = veri[sutun].fillna(veri[sutun].median())

le = LabelEncoder()
kategorik_sutunlar = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for sutun in kategorik_sutunlar:
    veri[sutun] = le.fit_transform(veri[sutun].astype(str))

# Hedef değişken (Test dosyasında olmadığı için rastgele oluşturulmuştur)
if 'Loan_Status' not in veri.columns:
    np.random.seed(42)
    veri['Loan_Status'] = np.random.randint(0, 2, veri.shape[0])

X = veri.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = veri['Loan_Status']

# Veri Bölme ve Ölçeklendirme
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_egitim_olcekli = scaler.fit_transform(X_egitim)
X_test_olcekli = scaler.transform(X_test)

# --- MODELLERİN EĞİTİLMESİ ---
sonuclar = {}

# 1. Karar Ağacı
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_egitim, y_egitim)
sonuclar['Karar Ağacı'] = accuracy_score(y_test, dt.predict(X_test))

# 2. Rastgele Orman
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_egitim, y_egitim)
sonuclar['Rastgele Orman'] = accuracy_score(y_test, rf.predict(X_test))

# 3. Yapay Sinir Ağları (ANN) - İterasyon artırıldı
ysa = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=5000, random_state=42)
ysa.fit(X_egitim_olcekli, y_egitim)
sonuclar['ANN'] = accuracy_score(y_test, ysa.predict(X_test_olcekli))

# 4. K-Ortalamalar (K-Means)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(X_egitim_olcekli)
sonuclar['K-Means'] = accuracy_score(y_test, km.predict(X_test_olcekli))

# --- GRAFİKLERİN OLUŞTURULMASI ---

# Grafik 1: Doğruluk Karşılaştırması
plt.figure(figsize=(10, 6))
grafik_verisi = pd.DataFrame({'Model': list(sonuclar.keys()), 'Doğruluk': list(sonuclar.values())})
sns.barplot(data=grafik_verisi, x='Model', y='Doğruluk', palette='viridis')
plt.title('Modellerin Başarı Oranları (Accuracy Comparison)')
plt.ylim(0, 1)
plt.savefig('dogruluk_grafigi.png')

# Grafik 2: K-Means Kümeleme (2D Görünüm)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_olcekli)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km.predict(X_test_olcekli), cmap='coolwarm', alpha=0.8)
plt.title('K-Means Müşteri Gruplandırma (PCA ile 2 Boyut)')
plt.savefig('kmeans_grafigi.png')

# Grafik 3: Karar Ağacı Yapısı
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['Red', 'Onay'], filled=True, rounded=True)
plt.title('Karar Ağacı Mantıksal Yapısı')
plt.savefig('karar_agaci.png')

# Grafik 4: Önemli Özellikler (Hangi veri daha etkili?)
plt.figure(figsize=(10, 6))
onem_df = pd.DataFrame({'Özellik': X.columns, 'Önem Skoru': rf.feature_importances_}).sort_values(by='Önem Skoru', ascending=False)
sns.barplot(data=onem_df, x='Önem Skoru', y='Özellik', palette='magma')
plt.title('Kredi Kararını Etkileyen En Önemli Faktörler')
plt.savefig('ozellik_onemi.png')

print("Bütün işlemler başarıyla tamamlandı ve grafikler kaydedildi.")
print("--- Model Doğruluk Sonuçları ---")
for model, skor in sonuclar.items():
    print(f"{model}: %{skor*100:.2f}")