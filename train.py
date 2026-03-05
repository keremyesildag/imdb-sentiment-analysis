"""
Sentiment Analysis - Film Yorumu Duygu Analizi
Kullanılan: scikit-learn, TF-IDF, Logistic Regression
Dataset : Kaggle — IMDB Dataset of 50K Movie Reviews
         https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

# ── Veri ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/IMDB Dataset.csv"   # CSV indirdiysen
# CSV_PATH = "data/IMDB Dataset.xlsx"  # Excel indirdiysen → bu satırı aç, üsttekini kapat


def clean_text(text: str) -> str:
    """HTML tag ve fazla boşlukları temizle."""
    text = re.sub(r"<[^>]+>", " ", text)   # <br />, <b> vs.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(path: str = CSV_PATH, sample: int = None):
    """
    Kaggle IMDB dosyasını yükle (CSV veya Excel).
    - path   : Dosyanın konumu (.csv, .xlsx veya .xls)
    - sample : Hızlı test için satır sayısı sınırı (None = tüm veri)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        print(f"  Excel okunuyor: {path}")
        df = pd.read_excel(path)
    else:
        print(f"  CSV okunuyor: {path}")
        df = pd.read_csv(path)

    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"  Örnekleme yapıldı: {sample} yorum")

    df["review"] = df["review"].apply(clean_text)
    df["label"]  = (df["sentiment"] == "positive").astype(int)

    pos = df["label"].sum()
    neg = len(df) - pos
    print(f"  Toplam: {len(df):,} yorum  |  Pozitif: {pos:,}  Negatif: {neg:,}")
    return df["review"].tolist(), df["label"].tolist()


# ── Model ────────────────────────────────────────────────────────────────────
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ])


# ── Değerlendirme ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negatif", "Pozitif"],
                yticklabels=["Negatif", "Pozitif"], ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix kaydedildi: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"ROC Curve (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Eğrisi")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ ROC eğrisi kaydedildi: {save_path}")


# ── Ana Akış ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   🎬  Duygu Analizi — Film Yorumları")
    print("=" * 55)

    # 1. Veri
    print("\n📂 Veri yükleniyor...")
    # Tüm 50K veri için: load_data()
    # Hızlı test için  : load_data(sample=5000)
    reviews, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"  Eğitim: {len(X_train)} örnek  |  Test: {len(X_test)} örnek")

    # 2. Eğitim
    print("\n🏋️  Model eğitiliyor...")
    model = build_pipeline()
    model.fit(X_train, y_train)
    print("  ✓ Eğitim tamamlandı.")

    # 3. Değerlendirme
    print("\n📊 Sonuçlar:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["Negatif", "Pozitif"]))

    # 4. Grafikler
    print("📈 Grafikler oluşturuluyor...")
    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, "outputs/confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, "outputs/roc_curve.png")

    # 5. Model Kaydet
    print("\n💾 Model kaydediliyor...")
    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("  ✓ models/sentiment_model.pkl")

    # 6. Demo Tahmin
    print("\n🔍 Demo Tahminler:")
    demo_texts = [
        "This movie was absolutely incredible and inspiring!",
        "Terrible film. Boring and a complete waste of time.",
        "Great performances but the plot was a bit slow.",
    ]
    for text in demo_texts:
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0][pred]
        label = "😊 Pozitif" if pred == 1 else "😞 Negatif"
        print(f"  [{label}  {prob:.0%}]  {text[:55]}...")

    print("\n✅ Tamamlandı!\n")


if __name__ == "__main__":
    main()
