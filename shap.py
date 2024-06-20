# SHAP interpretation
class BertForSequenceClassificationWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input texts should be a string or a list of strings.")
        
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=126, return_tensors='pt')
        encodings = {key: val.to(self.device) for key, val in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        return probs

# Initialize SHAP explainer
explainer_model = BertForSequenceClassificationWrapper(model, tokenizer, device)
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(explainer_model.predict, masker)



# Select a sample to explain
idx = 0  # Change this to the index of the sample you want to explain
sample_text = X_test.iloc[idx]
sample_text = sample_text[:512]
sample_text = explainer([sample_text])
print(f"Sample text for SHAP explanation: {sample_text}")


predictor = LIMExplainer(model, tokenizer) # Custom LIME predictor
label_names = ["Positive", "Neutral", "Negative"] # Labels for LIME
explainer = LimeTextExplainer(class_names=label_names, split_expression=predictor.split_string) # Init LIME predictor

to_use = sample_text[-2:] # Select last two samples for prediction
for i, example in enumerate(to_use):
    logging.info(f"Example {i+1}/{len(to_use)} start") # Log start of explanation
    temp = predictor.split_string(example)
    # Check the document size
    doc_size = len(temp)
    if doc_size < 2:  # Adjust the threshold as needed
        logging.warning(f"Skipping example {i+1} due to insufficient document size.")
        continue
    
    try:
        exp = explainer.explain_instance(text_instance=example, classifier_fn=predictor.predict, num_features=doc_size)
        logging.info(f"Example {i + 1}/{len(to_use)} done")
        words = exp.as_list()
        # exp.local_exp normalization if needed
        # sum_ = 0.6
        # exp.local_exp = {x: [(xx, yy / (sum(hh for _, hh in exp.local_exp[x])/sum_)) for xx, yy in exp.local_exp[x]] for x in exp.local_exp}
        exp.show_in_notebook(text=True, labels=(exp.available_labels()[0],))
    except Exception as e:
        logging.error(f"Error explaining example {i+1}: {e}")

# # # Ensure the sample text is truncated to the max length
# # sample_text = sample_text[:512]

# # # Convert sample_text to a list containing the single text for SHAP
# # shap_values = explainer([sample_text])

# # # Visualize the explanation
# # shap.plots.text(shap_values[0])

# # # Visualize the explanation
# # shap_html = shap.plots.text(shap_values[0], display=False)

# # # Save the SHAP explanation to an HTML file
# # with open("shap_explanation.html", "w") as file:
# #     file.write(shap_html)

# # print("SHAP explanation saved to shap_explanation.html")


# # Word-level analysis
# shap_values_word = explainer([sample_text[:512]])
# shap_word_html = shap.plots.text(shap_values_word[0], display=False)
# with open("shap_explanation_word.html", "w") as file:
#     file.write(shap_word_html)
# print("Word-level SHAP explanation saved to shap_explanation_word.html")

# # Visualization is instance by instance
# # But getting token score for each word is possible
# # Only keep words with score higher than threshold
# # Analyze what these words are

# # Sentence-level analysis
# sentences = sample_text.split('. ')
# sentences = ['[CLS] ' + sentence for sentence in sentences]
# shap_values_sentence = explainer(sentences)
# shap_sentence_html = shap.plots.text(shap_values_sentence[0], display=False)
# with open("shap_explanation_sentence.html", "w") as file:
#     file.write(shap_sentence_html)
# print("Sentence-level SHAP explanation saved to shap_explanation_sentence.html")


# Get representative examples with the top score (6) & lowest score (1)
# Plus correct and incorrect predictions
# See what contributes
# See if model is learning

