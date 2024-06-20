import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import random
import datasets
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from transformers import AutoModel, DataCollatorWithPadding
import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import logging

# Load CSV file into a DataFrame
df = pd.read_csv("persuasive_essays.csv")

# Take only the first 1000 data points
# df = df.head(1000)

# Extract text and essay scores
X = df["full_text"]
y = df["holistic_essay_score"]

# Split X_train into train and test subsets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to binary labels (with 4 as the cut-off for persuasiveness (scores 1-6))
y_train_binary = (y_train >= 4).astype(int)
y_test_binary = (y_test >= 4).astype(int)


# Define the pre-trained model name
model_name = "distilbert/distilbert-base-uncased"
# model_name = "bert-base-uncased"

# Call the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Max length
max_length = 512

# Encode the text for training data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')

# Encode the text for test data
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')


class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.astype(np.float32)  # Ensure labels are float32

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Ensure labels are flattened and converted to long tensor
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

# Convert our tokenized data into a torch Dataset
train_dataset = MakeTorchData(train_encodings, y_train_binary.ravel())
valid_dataset = MakeTorchData(test_encodings, y_test_binary.ravel())

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Setting num_labels to 2
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
