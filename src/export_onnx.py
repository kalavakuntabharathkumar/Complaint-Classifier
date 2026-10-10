import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/distilbert")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    sample = tokenizer("I need help with a recent account charge.", return_tensors="pt", truncation=True)
    output = Path(args.model) / "model.onnx"
    torch.onnx.export(
        model, (sample["input_ids"], sample["attention_mask"]), str(output),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0:"batch", 1:"sequence"},
                      "attention_mask": {0:"batch", 1:"sequence"},
                      "logits": {0:"batch"}},
        opset_version=17,
    )
    print(f"Exported {output}")

if __name__ == "__main__":
    main()
