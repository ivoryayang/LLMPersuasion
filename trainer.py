import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import os
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import PersuasionDataset

# Load the datasets from .pt files
def load_datasets(train_path, valid_path):
    try:
        train_dataset = torch.load(train_path)
        valid_dataset = torch.load(valid_path)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    return train_dataset, valid_dataset

# Compute various classification metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': (predictions == labels).mean(),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro')
    }

# Callback for saving outputs during training
class SaveOutputCallback(TrainerCallback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        output_file = os.path.join(self.output_dir, f'logs_epoch_{state.epoch}.txt')
        with open(output_file, 'a') as f:
            for key, value in logs.items():
                f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on processed data")
    parser.add_argument('-train_path', type=str, help='Path to the training data file')
    parser.add_argument('-validation_path', type=str, help='Path to the validation data file')
    parser.add_argument('-base_model_name', type=str, default='distilbert-base-uncased', help='base model to be trained')
    parser.add_argument('-model_save_path', type=str, help='Directory to save the trained model')

    args = parser.parse_args()

    train_dataset, valid_dataset = load_datasets(args.train_path, args.validation_path)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    # Initialize and run trainer
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_name, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[SaveOutputCallback('./output_logs')]
    )
    trainer.train()
    trainer.evaluate()
