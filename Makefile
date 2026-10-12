.PHONY: prepare baseline train evaluate export app
prepare:
	python -m src.prepare_data
baseline:
	python -m src.train_baseline
train:
	python -m src.train_transformer
evaluate:
	python -m src.evaluate
export:
	python -m src.export_onnx
app:
	streamlit run app.py
