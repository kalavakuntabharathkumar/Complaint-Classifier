from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from .prepare_data import TEXT_COL

def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(train[TEXT_COL], train["label_id"])
    pred = model.predict(test[TEXT_COL])
    print("Macro-F1:", round(f1_score(test["label_id"], pred, average="macro"), 4))
    print(classification_report(test["label_id"], pred, zero_division=0))
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/tfidf_logreg.joblib")

if __name__ == "__main__":
    main()
