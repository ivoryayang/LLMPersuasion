from torch.utils.data import DataLoader  # DataLoader class for loading data
from sklearn.model_selection import train_test_split  # Function to split datasets
from transformers import AutoTokenizer  # Import AutoTokenizer for tokenizing text data
import logging  # Import logging to log data and debugging information
from data_handling import MakeTorchData

# Function to load data from a specified CSV file
def load_data(input_path):
    df = pd.read_csv(input_path)  # Load the CSV file into a DataFrame
    X = df["full_text"]  # Extract the text column for processing
    y = df["holistic_essay_score"]  # Extract the essay score column for labels
    return X, y

# Function to prepare training and validation datasets
def prepare_datasets(X, y, tokenizer, test_size=0.2, random_state=42, max_length=512):
    y_binary = (y >= 4).astype(int)  # Convert essay scores to binary labels
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=test_size, random_state=random_state)  # Split data
    # Tokenize the training data
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    # Tokenize the testing data
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    train_dataset = MakeTorchData(train_encodings, y_train.ravel())  # Wrap training data in custom Dataset class
    valid_dataset = MakeTorchData(test_encodings, y_test.ravel())  # Wrap validation data in custom Dataset class
    return train_dataset, valid_dataset

# Function to save the prepared datasets to disk
def save_datasets(train_dataset, valid_dataset, train_path, valid_path):
    torch.save(train_dataset, train_path)  # Save the training dataset using PyTorch's save function
    torch.save(valid_dataset, valid_path)  # Save the validation dataset using PyTorch's save function

# Main function to execute preprocessing
def main(input_path, model_name="distilbert-base-uncased", train_path='train_dataset.pt', valid_path='valid_dataset.pt'):
    X, y = load_data(input_path)  # Load data from the CSV file
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer from Hugging Face's transformers
    train_dataset, valid_dataset = prepare_datasets(X, y, tokenizer)  # Prepare datasets
    save_datasets(train_dataset, valid_dataset, train_path, valid_path)  # Save datasets to disk
    print(f"Datasets prepared and saved to {train_path} and {valid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and tokenize persuasive essay data')
    parser.add_argument('input_path', type=str, help='Path to input CSV file')  # Define command-line argument for input file path
    args = parser.parse_args()  # Parse command-line arguments
    main(args.input_path)  # Execute the main function with the provided input path
