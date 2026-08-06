# Titanic Hayatta Kalma Tahmini — Karşılaştırmalı Makine Öğrenmesi Raporu

## 1. Giriş ve Proje Amacı
Bu projenin amacı, Titanic felaketindeki yolcu bilgilerini kullanarak kişilerin hayatta kalıp kalmadığını ikili sınıflandırma (binary classification) algoritmaları ile tahmin etmektir. Odak noktası sadece Kaggle skorunu yükseltmek değil; farklı makine öğrenmesi tekniklerinin aynı veri ve değerlendirme koşulları altındaki davranışlarını, güçlü/zayıf yönlerini analiz etmektir.

---

## 2. Veri Ön İşleme ve Öznitelik Mühendisliği (Feature Engineering)
- **Eksik Veri Doldurma (Imputation):**
  - `Age` değişkeni, yolcuların isimlerinden çıkarılan unvanlara (`Title`: Mr, Mrs, Miss, Master vb.) göre gruplanarak gruptaki medyan yaş ile dolduruldu.
  - `Embarked` mod (en sık tekrar eden liman) ile, `Fare` ise bilet sınıfı medyanı ile dolduruldu.
- **Türetilen Yeni Özellikler:**
  - **`Title`**: İsimlerden unvanlar çekildi, nadir unvanlar 'Rare' altında toplandı.
  - **`FamilySize` & `IsAlone`**: `SibSp + Parch + 1` formülüyle aile büyüklüğü türetildi.
  - **`Has_Cabin`**: %77'si eksik olan `Cabin` sütunu ikili değişkene (1: var / 0: yok) dönüştürüldü.
- **Ölçekleme ve Encoding:**
  - `Sex` 0/1 şeklinde etiketlendi. `Embarked` ve `Title` One-Hot Encoding'e tabi tutuldu.
  - Mesafe/Margin tabanlı modeller için `StandardScaler` uygulandı. Bilet ücretindeki çarpıklık `log1p` dönüşümü ile düzeltildi.

---

## 3. Modelleme ve Değerlendirme Metodolojisi
- **Çapraz Doğrulama:** Veri boyutunun küçüklüğü nedeniyle sınıf oranlarını koruyan **Stratified 5-Fold Cross Validation** kullanıldı.
- **Değerlendirme Metrikleri:** Sınıf dengesizliği göz önüne alınarak modeller Accuracy, Precision, Recall, F1-Score ve ROC-AUC ile değerlendirildi.
- **Denenen Algoritmalar:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes, MLP (Neural Network).

---

## 4. Karşılaştırmalı Sonuçlar ve Yorumlar

### En Başarılı Modeller (Özet Tablo):
| Model | Val Accuracy | F1-Score | ROC-AUC | Eğitim Süresi (sn) |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting (Tuned)** | **0.8451** | **0.7856** | **0.8805** | ~0.31 |
| **Random Forest (Tuned)** | 0.8384 | 0.7769 | 0.8714 | ~1.12 |
| **Logistic Regression (Tuned)** | 0.8294 | 0.7748 | 0.8758 | ~0.08 |

### Technical Takeaways (Teknik Çıkarımlar):
1. **Ensemble (Topluluk) Yöntemlerin Üstünlüğü:** Ağaç bazlı Boosting yöntemi (`Gradient Boosting`), verideki karmaşık ve doğrusal olmayan ilişkileri en iyi öğrenen model oldu.
2. **Tekil Ağaçların Overfitting Eğilimi:** Yalın `Decision Tree` modeli kısıtlama uygulanmadığında eğitim setini ezberleme göstererek doğrulama setinde en düşük performanslardan birini verdi.
3. **Ölçeklemenin Önemi:** Mesafe ve çizgi tabanlı modeller (`KNN`, `Logistic Regression`, `SVM`), `StandardScaler` sonrası ciddi performans artışı sağladı.
4. **Veri Boyutu ve Yapay Sinir Ağları:** `MLP` (Yapay Sinir Ağı), bu boyuttaki (891 satır) tabüler verilerde yakınsama uyarıları verdi ve Ensemble ağaç yöntemlerinin gerisinde kaldı. Tabüler veride ensemble yöntemlerin başarısı bir kez daha doğrulandı.

---

## 5. Sonuç
Proje başarıyla tamamlanmış, uçtan uca modüler bir pipeline (`main.py`) kurulmuştur. Titanic veri setinde en yüksek başarıyı **%84.5 Accuracy** ve **0.7856 F1-Score** ile **Gradient Boosting** modeli vermiştir.