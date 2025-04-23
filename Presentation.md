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





