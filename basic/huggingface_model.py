# Pre-trained HF model implementation

# Michał's Component: Basic requirements implementation
from core.base_model import BaseModel
from transformers import pipeline

class HuggingFaceModel(BaseModel):
    """Pre-trained model implementation (Michał's responsibility)"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model = pipeline("text-classification", model=model_name)
    
    def predict(self, input_text: str) -> str:
        return self.model(input_text)[0]["label"]  # Michał's implementation