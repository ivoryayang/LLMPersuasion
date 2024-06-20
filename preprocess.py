import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Define a custom Dataset class to manage tokenized data for PyTorch model training
class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Store the encoded texts
        self.labels = labels.astype(np.float32)  # Convert labels to float32 for compatibility with PyTorch

    def __getitem__(self, idx):
        # Return a single tokenized input and its label as a tensor
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()  # Convert labels to long tensor for classification
        return item

    def __len__(self):
        # Return the length of the dataset
        return len(self.labels)

# Function to load data from a CSV file
def load_data(input_path):
    df = pd.read_csv(input_path)  # Load data into DataFrame
    X = df["full_text"]  # Extract column with text data
    y = df["holistic_essay_score"]  # Extract column with essay scores
    return X, y

# Function to prepare datasets for training and validation
def prepare_datasets(X, y, tokenizer, test_size=0.2, random_state=42, max_length=512):
    y_binary = (y >= 4).astype(int)  # Convert scores to binary labels, with 4 as the cut-off for persuasiveness
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=test_size, random_state=random_state)  # Split data

    # Tokenize the training data
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    # Tokenize the testing data
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    # Create dataset objects for training and validation
    train_dataset = MakeTorchData(train_encodings, y_train.ravel())
    valid_dataset = MakeTorchData(test_encodings, y_test.ravel())
    return train_dataset, valid_dataset

# Main function to setup and run the data preparation and model loading
def main(input_path, model_name="distilbert-base-uncased"):
    # Load data
    X, y = load_data(input_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Initialize tokenizer
    train_dataset, valid_dataset = prepare_datasets(X, y, tokenizer)  # Prepare datasets

    # Load a pre-trained sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1})
    print("Setup complete, model loaded and datasets prepared.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and tokenize persuasive essay data')
    parser.add_argument('input_path', type=str, help='Path to input CSV file')  # Command line argument for input file path
    args = parser.parse_args()
    
    main(args.input_path)  # Run main function with the provided input path
