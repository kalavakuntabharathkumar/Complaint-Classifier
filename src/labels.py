LABELS = [
    "Credit card or prepaid card",
    "Debt collection",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Checking or savings account",
    "Mortgage",
    "Vehicle loan or lease",
    "Student loan",
    "Money transfer, virtual currency, or money service",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
