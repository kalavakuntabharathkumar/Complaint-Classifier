import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .labels import LABEL2ID

TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"

def clean_text(value):
    text = str(value).replace("\\r", " ").replace("\\n", " ").strip()
    return " ".join(text.split())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/complaints.csv")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--limit", type=int, default=60000)
    args = parser.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].map(clean_text)
    df = df[(df[TEXT_COL].str.len() >= 20) & df[LABEL_COL].isin(LABEL2ID)]
    df = df.drop_duplicates(subset=[TEXT_COL]).head(args.limit).copy()
    df["label_id"] = df[LABEL_COL].map(LABEL2ID)
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"Saved {len(train)} train and {len(test)} test rows to {out}")
    print(df[LABEL_COL].value_counts().to_string())

if __name__ == "__main__":
    main()
