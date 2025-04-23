# Company Classifier

## Problem Statement
Classify a list of companies into one or more insurance-related categories using a new given taxonomy. This is a multi-label classification task.
The final submitted solution uses a hybrid rule-based approach (see main.py) that performed best in this context.

---

## Setup Instructions

After download the project locally, follow these steps for running:

### 1. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main script (hybrid model)
```bash
python main.py
```

This will classify companies and generate the `ml_insurance_challenge_annotated.csv` file as output.

---

## Notes
- `train_logistic_regression.py` and `train_setfit.py` represent experimental approaches.
- For any list of companies, running `main.py` should provide the label classification for most of the companies.



