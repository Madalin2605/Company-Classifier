# Solution Presentation

## Overview

This project addresses the multi-label classification of insurance companies into a predefined taxonomy, based on a given textual description. The goal was to develop a system that correctly assigns one or more labels to each entry.

---

## Approaches

### 1. Hybrid Rule-Based Approach with (`main.py` file)

- This is the final and best-performing solution.
- I used keyword-based matching on preprocessed company descriptions to identify relevant insurance labels.
- **Preprocessing steps included**:
  - Converting all text to lowercase
  - Removing special characters and punctuation
  - Combining multiple columns into a single text field
  - Normalizing text for consistent pattern matching
- From the 9,495 companies in the dataset, 5,640 were successfully labeled through this rule-based process.
- For another 1,000 companies, I leveraged a pre-trained model by MoritzLaurer (`MoritzLaurer/xtremedistil-l6-h384-uncased`) to predict additional labels.
  - Only predictions with a confidence score above 0.5 were kept.
  - Inference was limited to 1,000 entries due to processing time constraints.
  - Validity of predicted labels was checked by ensuring they belonged to the same semantic category as the rule-based labels.

### 2. Logistic Regression Classifier
- Implemented using `scikit-learn` with `MultiLabelBinarizer`.
- I used a small labeled dataset generated with the hybrid rule-based method.
- Logistic regression was chosen due to its simplicity and effectiveness in multi-label classification.
- However, the model struggled to generalize because of:
  - The small size of the training set
  - Significant label imbalance across examples
- As a result, the model achieved low performance, with micro/macro F1-scores between 0.1 and 0.15.

### 3. SetFit Model Fine-Tuning
- Fine-tuned `paraphrase-MiniLM-L6-v2` using the `SetFit` framework on the same dataset as the logistic regression model.
- Achieved improved classification quality over logistic regression.
- Performance metrics, with micro/macro F1-scores between 0.38 and 0.55.
- Observations:
  - Better label assignment quality and generalization than logistic regression
  - However, a significant number of companies received no predicted labels











