import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset, DatasetDict

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

df['insurance_label'] = df['insurance_label'].apply(eval)

all_labels = [label for labels in df['insurance_label'] for label in labels]
label_counts = pd.Series(all_labels).value_counts()
valid_labels = set(label_counts[label_counts >= 5].index)

df['insurance_label'] = df['insurance_label'].apply(lambda labels: [l for l in labels if l in valid_labels])
df = df[df['insurance_label'].map(len) > 0]

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    df['full_text'], df['insurance_label'], test_size=0.2, random_state=42
)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_raw)
y_test = mlb.transform(y_test_raw)

train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2", multi_target_strategy="one-vs-rest")

trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    batch_size=16,
    num_iterations=20,
    num_epochs=10,
    column_mapping={"text": "text", "label": "label"},
)

trainer.train()

y_pred = trainer.model.predict(X_test.tolist())

micro_f1 = f1_score(y_test, y_pred, average="micro")
print("Micro F1:", micro_f1)

macro_f1 = f1_score(y_test, y_pred, average="macro")
print("Macro F1:", macro_f1)