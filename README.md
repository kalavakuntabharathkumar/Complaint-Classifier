# Complaint Classifier with Fine-Tuned DistilBERT

An end-to-end text classification project for assigning CFPB consumer complaint narratives to eight product categories. It includes a reproducible preprocessing pipeline, TF-IDF/logistic-regression baseline, DistilBERT fine-tuning, per-class evaluation, ONNX export and a Streamlit triage interface.

## Dataset
Download the public CFPB Consumer Complaint Database CSV from the CFPB website and save it as `data/complaints.csv`. The pipeline expects `Consumer complaint narrative` and `Product` columns. Keep the original source and download date in your experiment notes. Do not commit the raw dataset.

The default label map selects eight common categories. Rows without a narrative or outside the selected categories are removed. The script uses a stratified 80/20 split and saves split indices to support reproducibility.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_data --input data/complaints.csv
python -m src.train_baseline
python -m src.train_transformer
python -m src.evaluate --model artifacts/distilbert
python -m src.export_onnx --model artifacts/distilbert
streamlit run app.py
```

Training a transformer requires a suitable GPU for practical iteration. CPU execution is supported but slower.

## Metrics
The project brief describes a macro-F1 increase from 0.71 to 0.80 and CPU inference reduction from 46 ms to 19 ms. These are experiment targets/reported values, not hard-coded outputs. Re-run training and benchmark on the exact 60,000-row filtered dataset and record hardware, library versions, seed and sequence length before publishing those claims.

## App
The Streamlit app accepts a complaint narrative and displays the predicted category and class probabilities. If ONNX artifacts are unavailable, it shows a clear setup message instead of silently loading a different model.
