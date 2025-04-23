import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# Load the dataset
df = pd.read_csv("training_attempts\dataset.csv")
df = df[df["insurance_label"].notna() & (df["insurance_label"] != "[]")]

# Combine the text columns into a single column
df["full_text"] = (
    df["description"] + " " +
    df["business_tags"] + " " +
    df["sector"] + " " +
    df["category"] + " " +
    df["niche"]
)

def preprocess(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

df["full_text"] = df["full_text"].apply(preprocess)
df = df[["full_text", "insurance_label"]]

print(df.head(3))
print("Rows with empty 'insurance_label' have been removed, and only 'full_text' and 'insurance_label' columns are retained.")

flat_labels = [label for sublist in df['insurance_label'] for label in sublist]
print("Most common labels BEFORE:", Counter(flat_labels))
print("\n")

label_counts = Counter(label for labels in df['insurance_label'] for label in labels)
print("Label counts:", label_counts)
print("\n")

# Remove labels with fewer than 5 occurrences
min_occurrence = 5
valid_labels = set(label for label, count in label_counts.items() if count >= min_occurrence)
for label, count in label_counts.items():
    if label in valid_labels:
        print(f"{label}: {count}")
print("\n")

print("Valid labels:", Counter(valid_labels))
print("\n")

def filter_labels(labels):
    return [label for label in labels if label in valid_labels]

df['insurance_label'] = df['insurance_label'].apply(filter_labels)
flat_labels = [label for sublist in df['insurance_label'] for label in sublist]
print("Most common labels AFTER:", Counter(flat_labels))
print("\n")

df = df[df['insurance_label'].map(len) > 0]

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    df['full_text'], df['insurance_label'], test_size=0.2, random_state=42
)

flat_labels_now = [label for sublist in y_train_raw for label in sublist]
print("Most common labels IN TRAIN:", Counter(flat_labels_now))
print("\n")

train_counts = Counter(label for labels in y_train_raw for label in labels)
print("Train counts: ", train_counts)
print("\n")

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_raw)
y_test = mlb.transform(y_test_raw)

# Create a pipeline with TfidfVectorizer and LogisticRegression
# and use OneVsRestClassifier for multi-label classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(solver='liblinear'))
])

classifier = OneVsRestClassifier(pipeline)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

print("Subset accuracy:", accuracy_score(y_test, y_pred))
print("Micro F1:", f1_score(y_test, y_pred, average='micro'))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))