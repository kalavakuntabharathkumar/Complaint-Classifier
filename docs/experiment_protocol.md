# Experiment protocol

- Dataset: CFPB complaints with non-empty narratives and the eight configured product labels.
- Split: stratified 80/20, fixed random seed 42; do not tune on the held-out test set.
- Baseline: word/bi-gram TF-IDF plus class-balanced logistic regression.
- Fine-tuned model: DistilBERT, max sequence length 256, AdamW defaults from Transformers Trainer, learning rate 2e-5.
- Primary metric: macro-F1; also inspect per-class precision/recall and confusion matrix.
- ONNX latency: warm up both implementations, run at least 1,000 single-record inferences on the same CPU, report median and p95 separately. Include tokenizer time or explicitly report model-only latency.
- The stated 0.71 to 0.80 macro-F1 and 46 ms to 19 ms latency are project brief results. Reproduce and retain logs before using them as verified claims.
