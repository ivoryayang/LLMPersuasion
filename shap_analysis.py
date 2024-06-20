import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import pandas as pd

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_path = './eb5/checkpoint-975'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to('cuda')  # Adjust to 'cpu' if no GPU available

# Load a sample of data for SHAP analysis
data_path = './persuasive_essays.csv'
data = pd.read_csv(data_path)
sample_texts = data['full_text'].tolist()[:10]  # Ensure this is a list of strings

# Ensure each text is indeed a string
sample_texts = [str(text) for text in sample_texts]

# Initialize SHAP explainer with the model and tokenizer
explainer = shap.Explainer(model, tokenizer)

# Compute SHAP values
try:
    shap_values = explainer(sample_texts)
    print("SHAP values computed successfully.")
    print("Type of SHAP values:", type(shap_values))
    if isinstance(shap_values, list):
        print("Length of SHAP values list:", len(shap_values))
        print("Type of first element in SHAP values list:", type(shap_values[0]))
        if hasattr(shap_values[0], 'values'):
            print("Shape of SHAP values in the first element:", shap_values[0].values.shape)
    else:
        print("Shape of SHAP values:", shap_values.shape if hasattr(shap_values, 'shape') else 'N/A')
except Exception as e:
    print(f"Error computing SHAP values: {e}")
    shap_values = None

# Check if shap_values was computed and visualize
if shap_values is not None:
    try:
        if isinstance(shap_values, list) and hasattr(shap_values[0], 'values'):
            shap.plots.text(shap_values[0])  # Visualize the first explanation
        else:
            print("Unsupported SHAP values structure for text visualization.")
    except Exception as e:
        print("Error in visualization:", e)
else:
    print("SHAP values were not computed due to an error.")
