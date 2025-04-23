# Veridion Insurance Classification Challenge

This project tackles the task of assigning one or more labels from an insurance taxonomy to a list of companies based on their descriptions.

The final submitted solution uses a **hybrid rule-based approach** (see `main.py`) that performed best in this context, while two additional training-based experiments are available under `training_attempts/`.

---

## 🔧 Setup Instructions

To run the project locally, follow these simple steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/veridion-assessment.git
cd veridion-assessment
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the main script (hybrid model)
```bash
python main.py
```

This will classify companies and generate the `ml_insurance_challenge_annotated.csv` file as output.

---

## 📁 Project Structure
```
├── data/                         # Input files (e.g., dataset.csv)
├── training_attempts/           # Logistic Regression and SetFit model training scripts
├── main.py                      # Hybrid rule-based classifier (final solution)
├── ml_insurance_challenge_annotated.csv   # Final output with predicted labels
├── requirements.txt             # Project dependencies
```

---

## 💬 Notes
- `train_logistic_regression.py` and `train_setfit.py` represent experimental approaches.
- Only labels with enough support in the training data were considered.
- For any list of companies, running `main.py` should provide the label classification.

---

Feel free to explore and adapt the solution!
