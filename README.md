# Company Classifier

## Problem Statement
Classify a list of companies into one or more insurance-related categories using a given taxonomy. This is a multi-label classification task.

---

## 🔍 Solution Overview

The project explores **three approaches**, each with their own folder and logic:

1. **Hybrid Rule-Based Classifier (Best performing - `main.py`)**
   - A lightweight, explainable solution built on handcrafted rules.
   - Leverages keywords and text patterns.
   - Performed best in terms of relevance and accuracy on the dataset.

2. **Training-Based Attempt: Logistic Regression (`training_attempts/train_logistic_regression.py`)**
   - Used TF-IDF vectorization + MultiLabelBinarizer + One-vs-Rest classifier.
   - Struggled with data sparsity and imbalanced label distribution.

3. **Training-Based Attempt: SetFit (`training_attempts/train_setfit.py`)**
   - Fine-tuned `sentence-transformers/paraphrase-MiniLM-L6-v2` with SetFit for multi-label.
   - Reasonable performance (Micro F1 ~ 0.54), but slightly underperformed vs hybrid method.

---

## 🔹 File Structure
```
VERIDION_ASSESSMENT/
│
├── data/                              # Contains the original input CSV and Excel
│   └── dataset.csv
│
├── training_attempts/                # Contains experiments with ML models
│   ├── train_logistic_regression.py
│   └── train_setfit.py
│
├── checkpoints/                      # Saved models from training (SetFit)
│
├── main.py                           # Final hybrid rule-based classifier
│
├── ml_insurance_challenge_annotated.csv   # Final output with predicted insurance_label column
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🎨 Highlights
- ✅ Used rule-based filtering and domain knowledge to create precise keyword-label associations.
- ✅ Evaluated with Micro and Macro F1 to measure model generalization.
- ✅ Saved model checkpoints and binarizer for reproducibility (SetFit).
- ✅ All approaches can classify any new list of companies.

---

## 🚀 Run the Project
### Setup:
```bash
pip install -r requirements.txt
```

### Run Final Classifier:
```bash
python main.py
```

This will output predictions to `ml_insurance_challenge_annotated.csv`.

---

## 🎓 Conclusion
Despite limited labeled data, the hybrid method performed best.
Machine learning-based solutions were explored and structured for scalability in case more labeled data becomes available.

---

## 🔗 Author
Diaconu Andrei-Mădălin

