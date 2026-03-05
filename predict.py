"""
Eğitilmiş modeli kullanarak yeni yorumları tahmin et.
Kullanım: python src/predict.py "Bu film harikaydı!"
"""

import sys
import pickle


def load_model(path="models/sentiment_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(text: str, model=None):
    if model is None:
        model = load_model()
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    label = "Pozitif 😊" if pred == 1 else "Negatif 😞"
    confidence = proba[pred]
    return {
        "text": text,
        "label": label,
        "confidence": f"{confidence:.1%}",
        "positive_prob": f"{proba[1]:.1%}",
        "negative_prob": f"{proba[0]:.1%}",
    }


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python src/predict.py \"Yorum metni buraya\"")
        print("\nÖrnek yorumlar ile test ediliyor...\n")
        samples = [
            "What an incredible masterpiece!",
            "Terrible and boring. Hated every minute.",
            "Pretty average film, nothing special.",
        ]
    else:
        samples = [" ".join(sys.argv[1:])]

    model = load_model()
    print(f"{'─'*55}")
    print(f"  {'METİN':<35} {'TAHMİN':<15} {'GÜVEN'}")
    print(f"{'─'*55}")
    for text in samples:
        result = predict(text, model)
        print(f"  {text[:33]:<35} {result['label']:<15} {result['confidence']}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
