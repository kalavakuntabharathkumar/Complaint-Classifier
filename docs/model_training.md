# Model training

`src/train_transformer.py` fine-tunes `distilbert-base-uncased` using the fixed prepared split. The script logs evaluation metrics at each epoch and saves the best checkpoint under `artifacts/distilbert`. For a defensible experiment, save the exact filtered dataset row count, class distribution, package versions, random seed, GPU/CPU details and full Trainer logs alongside each run.
