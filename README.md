# 🎬 Film Yorumu Duygu Analizi (Sentiment Analysis)

Scikit-learn kullanarak film yorumlarını **Pozitif / Negatif** olarak sınıflandıran NLP projesi.

## 🚀 Özellikler

- **TF-IDF Vektörizasyon** — unigram + bigram desteği
- **Logistic Regression** sınıflandırıcı
- **Pipeline** mimarisi (vektörizasyon + model tek adımda)
- Confusion Matrix ve ROC eğrisi görselleştirme
- Eğitilmiş modeli `.pkl` olarak kaydetme / yükleme
- Yeni yorumlar için hazır tahmin scripti

## 📁 Proje Yapısı

```
sentiment-analysis/
├── src/
│   ├── train.py       # Model eğitimi ve değerlendirme
│   └── predict.py     # Yeni yorumlar için tahmin
├── models/            # Kaydedilen modeller (.pkl)
├── outputs/           # Grafikler (confusion matrix, ROC)
├── requirements.txt
└── README.md
```

## ⚙️ Kurulum

```bash
git clone https://github.com/keremyesildag/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

## 🏋️ Model Eğitimi

```bash
python src/train.py
```

Çıktılar:
- Accuracy ve ROC-AUC skoru
- Classification Report
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `models/sentiment_model.pkl`

## 🔍 Tahmin Yapma

```bash
# Tek yorum
python src/predict.py "This movie was absolutely amazing!"

# Birden fazla kelime
python src/predict.py "Terrible film, complete waste of time"
```

Örnek çıktı:
```
───────────────────────────────────────────────────────
  METİN                               TAHMİN          GÜVEN
───────────────────────────────────────────────────────
  This movie was absolutely amazing!  Pozitif 😊      94.3%
───────────────────────────────────────────────────────
```

## 📊 Model Performansı

| Metrik    | Değer |
|-----------|-------|
| Accuracy  | ~0.92 |
| ROC-AUC   | ~0.97 |
| Precision | ~0.92 |
| Recall    | ~0.92 |

## 🛠️ Kullanılan Teknolojiler

| Kütüphane    | Amaç                      |
|--------------|---------------------------|
| scikit-learn | Model & vektörizasyon      |
| pandas       | Veri işleme               |
| numpy        | Sayısal hesaplamalar       |
| matplotlib   | Görselleştirme             |
| seaborn      | İstatistiksel grafikler    |

## 📚 Nasıl Çalışır?

1. **TF-IDF**: Her yorumdaki kelimeleri sayısal vektörlere dönüştürür
2. **Logistic Regression**: Bu vektörlerden pozitif/negatif sınıfı öğrenir
3. **Pipeline**: Tahmin sırasında otomatik olarak aynı dönüşümleri uygular

## 🔧 Geliştirme Fikirleri

- [ ] Daha büyük IMDB veri seti ile yeniden eğit
- [ ] Türkçe duygu analizi desteği
- [ ] Flask/FastAPI ile REST API
- [ ] BERT gibi transformer modellere geçiş

## 📄 Lisans

MIT License
