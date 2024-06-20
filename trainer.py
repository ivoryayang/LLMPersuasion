import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import DataLoader
import os
import argparse

# Load the datasets from .pt files
def load_datasets(train_path, valid_path):
    train_dataset = torch.load(train_path)
    valid_dataset = torch.load(valid_path)
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
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Save model output logs at each epoch
        output_file = os.path.join(self.output_dir, f'epoch_{state.epoch}.txt')
        with open(output_file, 'w') as f:
            f.write(f'Logs saved for epoch {state.epoch}')

def main(train_path, valid_path, model_save_path):
    # Load data
    train_dataset, valid_dataset = load_datasets(train_path, valid_path)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    # Initialize and run trainer
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on processed data")
    parser.add_argument('train_path', type=str, help='Path to the training data file')
    parser.add_argument('valid_path', type=str, help='Path to the validation data file')
    parser.add_argument('model_save_path', type=str, help='Directory to save the trained model')
    args = parser.parse_args()
    
    main(args.train_path, args.valid_path, args.model_save_path)
