# Müşteri Ayrılma (Churn) Tahmini

Türkiye Yapay Zeka Akademisi — Makine Öğrenmesi Ara Ödevi

## Projenin Amacı

Bu proje, derste işlenen temel makine öğrenmesi akışını küçük ve anlaşılır bir
sınıflandırma problemi üzerinde uygulamayı amaçlar. Senaryo: bir aboneliğe
dayalı hizmetin müşterilerinin, elimizdeki bilgilere (yaş, gelir, abonelik
süresi, destek talebi sayısı, şehir, üyelik tipi) bakılarak hizmeti bırakıp
bırakmayacağının (churn) tahmin edilmesi.

Ödev kapsamında gerçek bir veri seti sağlanmadığı için, veri seti proje
içinde Python ile (NumPy'nin rastgele sayı üreteci kullanılarak, sabit bir
seed ile tekrarlanabilir şekilde) sentetik olarak oluşturulmuştur. `churn`
etiketi tamamen rastgele değildir: abonelik süresi kısa ve destek talebi
sayısı yüksek olan müşterilerde ayrılma olasılığı kasıtlı olarak
yükseltilmiştir, böylece modellerin öğrenebileceği gerçekçi bir örüntü
oluşturulmuştur.

## Proje Yapısı

```
.
├── musteri_ayrilma_tahmini.py   # Tüm akışın uygulandığı tek Python dosyası
├── requirements.txt             # Gerekli kütüphaneler
└── README.md
```

## Kullanılan Akış

1. Sentetik veri setinin oluşturulması (300 satır)
2. Veri setinin ilk incelemesi (satır/sütun sayısı, hedef değişken dağılımı)
3. Eksik değer kontrolü ve doldurma (sayısal → medyan, kategorik → mod)
4. Yeni özniteliklerin üretilmesi (`destek_talebi_var_mi`, `gelir_grubu`)
5. Kategorik değişkenlerin One-Hot Encoding ile sayısallaştırılması
6. Train (%60) / Validation (%20) / Test (%20) olarak stratified bölme
7. Sayısal değişkenlerin StandardScaler ile ölçeklenmesi (yalnızca train
   verisiyle fit edilip val/test'e transform uygulanır — veri sızıntısı
   önlenir)
8. Üç modelin eğitilmesi: Logistic Regression, KNN, (bonus) Decision Tree
9. Modellerin validation seti üzerinde F1-score'a göre karşılaştırılması
10. En iyi modelin test setinde confusion matrix, accuracy, precision,
    recall ve F1-score ile değerlendirilmesi
11. Kısa sonuç yorumu

## Nasıl Çalıştırılır

```bash
# (Opsiyonel ama önerilir) sanal ortam oluşturun
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Script'i çalıştırın
python musteri_ayrilma_tahmini.py
```

Script harici bir veri dosyasına ihtiyaç duymaz; veri seti çalıştırma
anında kod içinde üretilir. Tüm adımların çıktıları (tablolar, metrikler,
yorum) konsola sırasıyla yazdırılır.

## Kısa Sonuç Yorumu

Validation seti üzerindeki F1-score karşılaştırmasına göre en iyi
performansı **Logistic Regression** modeli göstermiştir (F1 ≈ 0.52) ve bu
nedenle test seti değerlendirmesi için bu model seçilmiştir. Test setinde
elde edilen sonuçlar (Accuracy ≈ 0.82, Precision ≈ 0.67, Recall ≈ 0.63,
F1-score ≈ 0.65), modelin hem ayrılan hem de kalan müşterileri makul bir
dengeyle ayırt edebildiğini göstermektedir.

Logistic Regression'ın KNN ve Decision Tree'ye kıyasla öne çıkmasının olası
nedeni, veri setindeki öznitelik-hedef ilişkisinin (abonelik süresi ve
destek talebi sayısına dayalı, doğrusala yakın bir ilişki olarak
kurgulanmış olması) az sayıda öznitelikle özetlenebilir olmasıdır. Küçük ve
orta ölçekli, aşırı karmaşık olmayan veri setlerinde basit ve doğrusal
modeller, karmaşıklığı yüksek modellere (örn. derin karar ağaçları) kıyasla
benzer ya da daha iyi genelleme performansı gösterebilir; bu proje küçük
ölçekte bu genel eğilimi de gözlemleme fırsatı sunmaktadır.

> Not: Veri seti her çalıştırmada aynı `random_state` (42) ile üretildiği
> için sonuçlar tekrarlanabilirdir; farklı bir seed veya farklı veri
> büyüklüğü ile metrik değerleri değişebilir.
