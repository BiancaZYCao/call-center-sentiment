from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load the fine-tuned model and tokenizer from the saved directory
model_path = './finetuned_distilbert_client_Oct'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()
# Define label mapping based on the classes your model was trained on
label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


def text_sentiment_inference(text):
    """
    Perform sentiment analysis on a single input text string and return the output class label.

    Parameters:
    text (str): The input text string to analyze.

    Returns:
    str: The predicted sentiment class label ('Negative', 'Neutral', or 'Positive').
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    # Perform inference using the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the predicted class label index
    predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
    # Map the predicted index to the corresponding class label
    predicted_label = label_mapping[predicted_class_idx]
    return predicted_label

# Example usage (test when initialized)
example_text = "This is the worst travel experience I've ever had." 
predicted_sentiment = text_sentiment_inference(example_text)
print(f"[Test] Text: '{example_text}' -> Predicted Sentiment: {predicted_sentiment}")