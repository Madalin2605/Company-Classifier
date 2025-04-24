import pandas as pd
import re
import torch
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm

# Set device to GPU if available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1

# Load the data
companies_df = pd.read_csv("data\ml_insurance_challenge.csv")
taxonomy_df = pd.read_excel("data\insurance_taxonomy.xlsx")
companies_df = companies_df.fillna("")

taxonomy_labels = taxonomy_df["label"].dropna().unique().tolist()

# Preprocess the taxonomy labels
companies_df["full_text"] = (
    companies_df["description"] + " " +
    companies_df["business_tags"] + " " +
    companies_df["sector"] + " " +
    companies_df["category"] + " " +
    companies_df["niche"]
)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

companies_df["full_text"] = companies_df["full_text"].apply(preprocess)

# Create a dictionary to store the labels for each company
company_labels = defaultdict(list)


# Iterate through the DataFrame and check for matches with taxonomy labels
for idx, row in companies_df.iterrows():
    text = row["full_text"]
    for label in taxonomy_labels:
        label_words = preprocess(label).split()
        if all(word in text for word in label_words):
            company_labels[idx].append(label)

companies_df["insurance_label"] = companies_df.index.map(lambda i: company_labels.get(i, []))

# Check if there are any labels assigned
unlabeled_companies_df = companies_df[companies_df["insurance_label"].apply(len) == 0]
unlabeled_companies_df = unlabeled_companies_df.head(1000)  # Limit to 1000 rows for zero-shot classification

# If there are any unlabeled rows, use zero-shot classification
classifier = pipeline(
    "zero-shot-classification",
    # model="facebook/bart-large-mnli",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v1",
    device=0
)
    
for idx, row in tqdm(unlabeled_companies_df.iterrows(), total=len(unlabeled_companies_df), desc="Zero-shot classification"):
    result = classifier(row["full_text"], candidate_labels=taxonomy_labels, multi_label=True)
    top_labels = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]
    companies_df.at[idx, "insurance_label"] = top_labels

# Convert the insurance_label column to a string representation and save the CSV file
companies_df.drop(columns=["full_text"], inplace=True)
companies_df.to_csv("ml_insurance_challenge_annotated.csv", index=False)