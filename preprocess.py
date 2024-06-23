import argparse
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader  # DataLoader class for loading data
from sklearn.model_selection import train_test_split  # Function to split datasets
from transformers import AutoTokenizer  # Import AutoTokenizer for tokenizing text data
import logging  # Import logging to log data and debugging information
from utils import PersuasionDataset, get_feature_mask_from_offsets

# Function to load data from a specified CSV file
def load_data(input_path):
    df = pd.read_csv(input_path)  # Load the CSV file into a DataFrame
    #comment out the following line if not testing
    df = df[:20]
    X = df["full_text"]  # Extract the text column for processing
    y = df["holistic_essay_score"]  # Extract the essay score column for labels
    return X, y

# Function to prepare training and validation datasets
def prepare_datasets(X, y, tokenizer, threshold, test_size=0.2, random_state=42, max_length=512):
    y_binary = (y >= threshold).astype(int)  # Convert essay scores to binary labels
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=test_size, random_state=random_state)  # Split data
    y_train_tensor = torch.from_numpy(y_train.ravel()).long()
    y_test_tensor = torch.from_numpy(y_test.ravel()).long()
    # Tokenize data
    train_text = X_train

    ######NOTE BELOW THIS "tolist()" implementation is terribad!!!!! it's not memory-friendly!!!!"######
    ######left like this for now but should be fixed ASAP######################################
    train_tokenized = tokenizer(train_text.tolist(), truncation=True, padding=True, max_length=max_length,
                                return_tensors='pt', return_offsets_mapping=True) #need to return this to get feature maps for attributions later
    print(f'getting feature masks for training text...')
    train_feature_mask = torch.tensor([get_feature_mask_from_offsets(t) for t in tqdm(train_tokenized['offset_mapping'])], dtype=torch.long)

    validation_text = X_test
    validation_tokenized = tokenizer(validation_text.tolist(), truncation=True, padding=True, max_length=max_length,
                                return_tensors='pt',  return_offsets_mapping=True)
    print(f'getting feature masks for validation text...')
    validation_feature_mask = torch.tensor([get_feature_mask_from_offsets(t) for t in tqdm(validation_tokenized['offset_mapping'])], dtype=torch.long)

    train_text_raw = train_text.tolist()
    validation_text_raw = validation_text.tolist()
    train_dataset = PersuasionDataset(train_text_raw, train_tokenized, y_train.ravel(), train_feature_mask)  # Wrap training data in custom Dataset class
    validation_dataset = PersuasionDataset(validation_text_raw, validation_tokenized, y_test_tensor, validation_feature_mask)  # Wrap validation data in custom Dataset class

    return train_dataset, validation_dataset

# Function to save the prepared datasets to disk
def save_datasets(train_dataset, valid_dataset, train_path, valid_path):
    torch.save(train_dataset, train_path)  # Save the training dataset using PyTorch's save function
    torch.save(valid_dataset, valid_path)  # Save the validation dataset using PyTorch's save function

# Main function to execute preprocessing
def main(input_path, model_name="distilbert-base-uncased", train_path='train_dataset.pt', valid_path='valid_dataset.pt'):
    print(f"Datasets prepared and saved to {train_path} and {valid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and tokenize persuasive essay data')
    parser.add_argument('-input_path', type=str, help='Path to input CSV file')
    parser.add_argument('-base_model_name', type=str, default='distilbert-base-uncased', help='base model with which the given data is to be processed')
    parser.add_argument('-train_path', type=str, default='train_dataset.pt', help='Path to output torch train file')
    parser.add_argument('-validation_path', type=str, default='validation_dataset.pt', help='Path to output torch validation file')
    parser.add_argument('-persuasive_threshold', type=float, default=4, help='Threshold for converting to binary labels')
    parser.add_argument('-validation_size', type=float, default=0.2, help='percentage of total data reserved for validation')

    args = parser.parse_args()  # Parse command-line arguments

    #prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)  # Load tokenizer from Hugging Face's transformers
    X, y = load_data(args.input_path)  # Load data from the CSV file
    train_dataset, valid_dataset = prepare_datasets(X=X, y=y, tokenizer=tokenizer,
                                                    test_size=args.validation_size, threshold=args.persuasive_threshold)  # Prepare datasets

    #save dataset
    save_datasets(train_dataset, valid_dataset, args.train_path, args.validation_path)  # Save datasets to disk

