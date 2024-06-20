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

def compute_metrics_for_classification(eval_pred):
    logits, labels = eval_pred
#     print(labels)
    predictions = np.argmax(logits, axis=1)
    labels = labels.flatten()

    # Convert logits and labels to PyTorch tensors
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)

    # Calculate cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits_tensor, labels_tensor)

    # Calculate additional classification metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
#     f1 = f1_score(labels, predictions)

    f1_macro = f1_score(labels, predictions,average='macro')
    f1_micro = f1_score(labels, predictions,average='micro')


    return {"loss": loss.item(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro}


class SaveOutputCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    def on_epoch_end(self, args, state, control, **kwargs):
        # Extract the current epoch from the state
        epoch = state.epoch
        if state.log_history:
            # Extract output from Trainer state
            output = state.log_history[-1]  # Get the log history for the last step of the epoch
            # Save output to a text file
            output_file = os.path.join(self.output_dir, f'epoch_{epoch}_output.txt')
            with open(output_file, 'w') as f:
                f.write(str(output))
            print(f"Logs saved for epoch {epoch}")
        else:
            print(f"No logs recorded for epoch {epoch}")
# Define your TrainerCallback
save_output_callback = SaveOutputCallback('./eb5/output_logs')

num_epochs = 20

# Update training_args to include the new callback
training_args = TrainingArguments(
        output_dir='./eb5',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=33, #Reducing batch size generalizes better
    per_device_eval_batch_size=33,
    weight_decay=1e-2, #0.99 increasing weight_decay generalizes better
    learning_rate=2e-5,  #Reducing learning rate also generalizes better
    logging_dir='./logs',
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    evaluation_strategy="epoch",  # Ensure evaluation is done at the end of each epoch
    save_strategy="epoch",        # Ensure model checkpoints are saved at the end of each epoch
    logging_strategy="epoch",     # Ensure logging is done at the end of each epoch
#     evaluation_strategy="steps",  # Ensure evaluation is done at the end of each epoch
# #     save_strategy="epoch",        # Ensure model checkpoints are saved at the end of each epoch
#     eval_steps=1,
#     logging_steps=1,
#     logging_strategy="steps",     # Ensure logging is done at the end of each epoch
    report_to="none",
    push_to_hub=False,  # Disable uploading results to the Hub
)

# Call the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics_for_classification,
    callbacks=[save_output_callback]  # Add the callback to the Trainer
)
# Train the model 
trainer.train()
# Call the summary
trainer.evaluate()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
