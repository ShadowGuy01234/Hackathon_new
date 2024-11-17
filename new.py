from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("final_model")
tokenizer = AutoTokenizer.from_pretrained("final_model")

# Classify new documents
documents = [
    "AI in healthcare is revolutionary.",
    "Quarterly financial reports show significant growth.",
    "Emerging trends in blockchain technology."
]
inputs = tokenizer(documents, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
print(predictions)  # Prints predicted labels
