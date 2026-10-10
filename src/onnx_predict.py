from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from .labels import ID2LABEL

class ComplaintPredictor:
    def __init__(self, model_dir="artifacts/distilbert"):
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    def predict(self, text: str):
        encoded = self.tokenizer(text, return_tensors="np", truncation=True, max_length=256)
        inputs = {key: encoded[key].astype(np.int64) for key in ("input_ids", "attention_mask")}
        logits = self.session.run(["logits"], inputs)[0][0]
        shifted = logits - logits.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        idx = int(probs.argmax())
        return {"label": ID2LABEL[idx], "confidence": float(probs[idx]),
                "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}}
