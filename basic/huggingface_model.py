# Pre-trained HF model implementation
# Michał's Component: Basic requirements implementation
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch


class HuggingFaceModel:
    """Pre-trained model implementation (Michał's responsibility)"""

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions", use_gpu=True):
        self.model_name = model_name
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        self.use_gpu = use_gpu

    def load_model(self, force=False):
        if self.model_loaded and not force:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if  self.use_gpu:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map='auto',
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )

        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )

        self.model_loaded = True

    def predict(self, input_text: str) -> dict[str, float]:
        if not self.model_loaded:
            self.load_model()

        results = self.pipeline(input_text)
        best_entry = max(results[0], key=lambda x: x["score"])
        return {best_entry["label"]: best_entry["score"]}