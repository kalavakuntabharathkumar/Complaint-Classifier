import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from .prepare_data import TEXT_COL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/distilbert")
    args = parser.parse_args()
    test = pd.read_csv("data/processed/test.csv")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    ds = Dataset.from_pandas(test[[TEXT_COL, "label_id"]].rename(columns={"label_id":"labels"}))
    ds = ds.map(lambda b: tokenizer(b[TEXT_COL], truncation=True, max_length=256), batched=True)
    pred = Trainer(model=model).predict(ds)
    y_pred = np.argmax(pred.predictions, axis=1)
    y_true = test["label_id"].to_numpy()
    print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
