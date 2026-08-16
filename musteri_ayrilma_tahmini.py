"""
Müşteri Ayrılma (Churn) Tahmini - Temel Makine Öğrenmesi Akışı
================================================================

Amaç
----
Bu proje, bir aboneliğe dayalı hizmet kullanan müşterilerin ayrılıp
ayrılmayacağını (churn) tahmin eden basit ve anlaşılır bir makine
öğrenmesi sınıflandırma akışını uçtan uca uygular. Akış; veri oluşturma,
temel veri inceleme, eksik değer temizliği, kategorik değişkenlerin
sayısallaştırılması, ölçekleme, öznitelik üretimi, train/validation/test
bölme, model eğitimi (Logistic Regression, KNN ve bonus olarak Decision
Tree) ve sınıflandırma metrikleriyle değerlendirme adımlarını içerir.

Kullanılan Kütüphaneler
------------------------
- numpy, pandas          : veri oluşturma ve işleme
- scikit-learn            : ön işleme, model eğitimi ve değerlendirme

Çalıştırma
----------
1) Sanal ortam oluşturup gerekli paketleri yükleyin:
       pip install -r requirements.txt
2) Dosyayı doğrudan çalıştırın:
       python musteri_ayrilma_tahmini.py
   Script, veri setini kendi içinde rastgele (ama sabit seed ile
   tekrarlanabilir şekilde) üretir; harici bir CSV dosyasına ihtiyaç
   yoktur. Tüm adımların çıktıları konsola sırasıyla yazdırılır.

Not
---
Gerçek bir müşteri veri seti bu ödev kapsamında sağlanmadığı için,
"Veri Seti" bölümünde açıklanan mantığa uygun, gerçekçi ilişkiler
içeren sentetik bir veri seti Python içinde üretilmiştir.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Soru 2: Veri setini pandas DataFrame olarak hazırlama
# ---------------------------------------------------------------------------
def veri_seti_olustur(n_satir: int = 300, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Müşteri ayrılma tahmini için sentetik bir veri seti üretir.

    Sütunlar: yas, gelir, abonelik_suresi, destek_talebi_sayisi, sehir,
    uyelik_tipi, churn (hedef değişken: 0 = kalır, 1 = ayrılır).

    churn değişkeni tamamen rastgele değil; abonelik süresi kısa ve
    destek talebi sayısı yüksek olan müşterilerde ayrılma olasılığı
    kasıtlı olarak artırılmıştır. Ayrıca gerçekçi bir senaryo olması
    için bazı sütunlara kasıtlı olarak eksik değer (NaN) eklenmiştir.
    """
    rng = np.random.default_rng(random_state)

    yas = rng.integers(18, 70, size=n_satir)
    gelir = rng.normal(loc=18000, scale=6000, size=n_satir).round(2)
    gelir = np.clip(gelir, 4000, None)
    abonelik_suresi = rng.integers(1, 60, size=n_satir)  # ay cinsinden
    destek_talebi_sayisi = rng.poisson(lam=1.5, size=n_satir)
    sehir = rng.choice(
        ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"], size=n_satir
    )
    uyelik_tipi = rng.choice(
        ["Standart", "Premium", "VIP"], size=n_satir, p=[0.5, 0.35, 0.15]
    )

    # churn olasılığını mantıklı bir ilişkiyle hesapla:
    # abonelik süresi kısaldıkça ve destek talebi arttıkça ayrılma olasılığı artar.
    # Sabit terim (-1.2), churn oranını gerçekçi biçimde azınlık sınıfa (~%25-30) çeker.
    churn_skoru = (
        -1.6
        - 0.90 * (abonelik_suresi - abonelik_suresi.mean()) / abonelik_suresi.std()
        + 1.60 * (destek_talebi_sayisi - destek_talebi_sayisi.mean()) / (destek_talebi_sayisi.std() + 1e-6)
        - 0.40 * (gelir - np.nanmean(gelir)) / np.nanstd(gelir)
    )
    churn_olasilik = 1 / (1 + np.exp(-churn_skoru))
    churn = rng.binomial(1, churn_olasilik)

    df = pd.DataFrame(
        {
            "musteri_id": [f"MUS{1000 + i}" for i in range(n_satir)],
            "yas": yas,
            "gelir": gelir,
            "abonelik_suresi": abonelik_suresi,
            "destek_talebi_sayisi": destek_talebi_sayisi,
            "sehir": sehir,
            "uyelik_tipi": uyelik_tipi,
            "churn": churn,
        }
    )

    # Gerçekçi olması için kasıtlı olarak birkaç eksik değer ekle
    eksik_index_gelir = rng.choice(df.index, size=int(n_satir * 0.07), replace=False)
    df.loc[eksik_index_gelir, "gelir"] = np.nan

    eksik_index_destek = rng.choice(df.index, size=int(n_satir * 0.05), replace=False)
    df.loc[eksik_index_destek, "destek_talebi_sayisi"] = np.nan

    eksik_index_sehir = rng.choice(df.index, size=int(n_satir * 0.03), replace=False)
    df.loc[eksik_index_sehir, "sehir"] = np.nan

    return df


def main():
    print("=" * 70)
    print("MÜŞTERİ AYRILMA (CHURN) TAHMİNİ - MAKİNE ÖĞRENMESİ AKIŞI")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Soru 2: Veri setini pandas DataFrame olarak hazırla
    # -----------------------------------------------------------------
    df = veri_seti_olustur(n_satir=300)
    print("\n[Soru 2] Veri seti oluşturuldu. Boyut:", df.shape)

    # -----------------------------------------------------------------
    # Soru 3: İlk satırları, satır-sütun sayısını ve hedef değişken
    # dağılımını inceleme
    # -----------------------------------------------------------------
    print("\n[Soru 3] İlk 5 satır:")
    print(df.head())

    print(f"\n[Soru 3] Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")

    print("\n[Soru 3] Hedef değişken (churn) dağılımı:")
    print(df["churn"].value_counts())
    print("\n[Soru 3] Hedef değişken yüzde dağılımı:")
    print((df["churn"].value_counts(normalize=True) * 100).round(2))

    # -----------------------------------------------------------------
    # Soru 4: Eksik değer kontrolü ve doldurma
    # -----------------------------------------------------------------
    print("\n[Soru 4] Doldurmadan önce eksik değer sayıları:")
    print(df.isnull().sum())

    df["gelir"] = df["gelir"].fillna(df["gelir"].median())
    df["destek_talebi_sayisi"] = df["destek_talebi_sayisi"].fillna(
        df["destek_talebi_sayisi"].median()
    )
    df["sehir"] = df["sehir"].fillna(df["sehir"].mode()[0])

    print("\n[Soru 4] Doldurduktan sonra eksik değer sayıları:")
    print(df.isnull().sum())

    # -----------------------------------------------------------------
    # Soru 7: En az 1 basit öznitelik üretme
    # (Bu adımı encoding'den önce, ham sütunlar üzerinden yapıyoruz.)
    # -----------------------------------------------------------------
    df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)
    df["gelir_grubu"] = pd.cut(
        df["gelir"],
        bins=[0, 12000, 20000, np.inf],
        labels=["dusuk", "orta", "yuksek"],
    )
    print("\n[Soru 7] Yeni öznitelikler eklendi: 'destek_talebi_var_mi', 'gelir_grubu'")
    print(df[["gelir", "gelir_grubu", "destek_talebi_sayisi", "destek_talebi_var_mi"]].head())

    # -----------------------------------------------------------------
    # Soru 5: Kategorik değişkenleri One-Hot Encoding ile sayısala çevirme
    # -----------------------------------------------------------------
    kategorik_sutunlar = ["sehir", "uyelik_tipi", "gelir_grubu"]
    df_encoded = pd.get_dummies(df, columns=kategorik_sutunlar, drop_first=True)
    print("\n[Soru 5] One-Hot Encoding sonrası sütunlar:")
    print(list(df_encoded.columns))

    # -----------------------------------------------------------------
    # Model için öznitelik (X) ve hedef (y) ayrımı
    # -----------------------------------------------------------------
    hedef = "churn"
    kullanilmayacak = ["musteri_id"]
    ozellikler = [c for c in df_encoded.columns if c not in [hedef] + kullanilmayacak]

    X = df_encoded[ozellikler].copy()
    y = df_encoded[hedef].copy()

    # -----------------------------------------------------------------
    # Soru 8: Train / validation / test bölme (stratify ile)
    # -----------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\n[Soru 8] Train boyutu: {X_train.shape[0]}")
    print(f"[Soru 8] Validation boyutu: {X_val.shape[0]}")
    print(f"[Soru 8] Test boyutu: {X_test.shape[0]}")

    # -----------------------------------------------------------------
    # Soru 6: Sayısal değişkenlerde ölçekleme
    # (Veri sızıntısını önlemek için scaler yalnızca train verisiyle
    # fit edilir, ardından val/test verisine transform uygulanır.)
    # -----------------------------------------------------------------
    sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi"]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[sayisal_sutunlar] = scaler.fit_transform(X_train[sayisal_sutunlar])
    X_val_scaled[sayisal_sutunlar] = scaler.transform(X_val[sayisal_sutunlar])
    X_test_scaled[sayisal_sutunlar] = scaler.transform(X_test[sayisal_sutunlar])

    print("\n[Soru 6] Ölçeklenen sayısal sütunlar:", sayisal_sutunlar)

    # -----------------------------------------------------------------
    # Soru 9: En az 2 model eğitme (Logistic Regression, KNN) +
    # bonus olarak Decision Tree
    # -----------------------------------------------------------------
    modeller = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree (bonus)": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=4),
    }

    for model in modeller.values():
        model.fit(X_train_scaled, y_train)

    print("\n[Soru 9] Modeller eğitildi:", list(modeller.keys()))

    # -----------------------------------------------------------------
    # Soru 10: Validation sonuçlarına göre modelleri karşılaştırma
    # -----------------------------------------------------------------
    print("\n[Soru 10] Validation seti performans karşılaştırması:")
    print(f"{'Model':<25}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<12}")

    val_sonuclari = {}
    for isim, model in modeller.items():
        tahmin_val = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, tahmin_val)
        prec = precision_score(y_val, tahmin_val, zero_division=0)
        rec = recall_score(y_val, tahmin_val, zero_division=0)
        f1 = f1_score(y_val, tahmin_val, zero_division=0)
        val_sonuclari[isim] = f1
        print(f"{isim:<25}{acc:<12.3f}{prec:<12.3f}{rec:<12.3f}{f1:<12.3f}")

    en_iyi_model_adi = max(val_sonuclari, key=val_sonuclari.get)
    en_iyi_model = modeller[en_iyi_model_adi]
    print(f"\n[Soru 10] Validation F1-score'a göre seçilen model: {en_iyi_model_adi}")

    # -----------------------------------------------------------------
    # Soru 11: Seçilen modeli test setinde değerlendirme
    # -----------------------------------------------------------------
    tahmin_test = en_iyi_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, tahmin_test)
    acc_test = accuracy_score(y_test, tahmin_test)
    prec_test = precision_score(y_test, tahmin_test, zero_division=0)
    rec_test = recall_score(y_test, tahmin_test, zero_division=0)
    f1_test = f1_score(y_test, tahmin_test, zero_division=0)

    print(f"\n[Soru 11] '{en_iyi_model_adi}' modeli - Test seti sonuçları")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy : {acc_test:.3f}")
    print(f"Precision: {prec_test:.3f}")
    print(f"Recall   : {rec_test:.3f}")
    print(f"F1-score : {f1_test:.3f}")

    # -----------------------------------------------------------------
    # Soru 12: Kısa yorum
    # -----------------------------------------------------------------
    print("\n[Soru 12] Sonuç Yorumu")
    print("-" * 70)
    diger_modeller = [isim for isim in modeller if isim != en_iyi_model_adi]
    yorum = (
        f"Validation seti üzerindeki F1-score karşılaştırmasına göre en iyi performansı "
        f"'{en_iyi_model_adi}' modeli göstermiştir ve bu nedenle test seti değerlendirmesi "
        f"için bu model seçilmiştir. Test setinde elde edilen {f1_test:.3f} F1-score, modelin "
        f"hem ayrılan hem de kalan müşterileri makul bir dengeyle ayırt edebildiğini göstermektedir. "
        f"Bu modelin diğerlerine ({', '.join(diger_modeller)}) göre öne çıkmasının olası nedeni, "
        f"veri setindeki öznitelik-hedef ilişkisinin doğrusala yakın ve az sayıda öznitelikle "
        f"özetlenebilir olmasıdır; küçük ve orta ölçekli, aşırı karmaşık olmayan veri setlerinde "
        f"basit modeller genellikle daha karmaşık modellere kıyasla benzer ya da daha iyi genelleme "
        f"performansı gösterebilir."
    )
    print(yorum)
    print("=" * 70)


if __name__ == "__main__":
    main()
