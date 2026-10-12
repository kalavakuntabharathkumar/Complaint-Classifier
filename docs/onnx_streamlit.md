# ONNX inference and triage UI

Export the fine-tuned checkpoint with `python -m src.export_onnx`. The app uses ONNX Runtime's CPU execution provider and returns a predicted product label with normalized class probabilities. Benchmark with `python -m src.benchmark --runs 1000`; compare against the PyTorch path under identical preprocessing and hardware before reporting speedups.
