import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score
from .labels import LABELS, ID2LABEL, LABEL2ID
from .prepare_data import TEXT_COL

BASE_MODEL = "distilbert-base-uncased"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    def tokenize(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=256)
    train_ds = Dataset.from_pandas(train[[TEXT_COL, "label_id"]].rename(columns={"label_id":"labels"}))
    test_ds = Dataset.from_pandas(test[[TEXT_COL, "label_id"]].rename(columns={"label_id":"labels"}))
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro"),
                "accuracy": accuracy_score(labels, preds)}
    output = "artifacts/distilbert"
    training = TrainingArguments(
        output_dir="runs/distilbert",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=100,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training, train_dataset=train_ds, eval_dataset=test_ds,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    print(trainer.evaluate())
    Path(output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output)
    tokenizer.save_pretrained(output)

if __name__ == "__main__":
    main()
