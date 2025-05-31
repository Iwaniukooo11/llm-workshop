# LLM model implementation
from core.base_model import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch


class LLMModel(BaseModel):
    def __init__(
        self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", emotion_list=None
    ):
        self.model_name = model_name
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        if emotion_list is None:
            self.emotion_list = [  # jak ta lista jest za długa, LLM ma problemy
                "joy",
                "sadness",
                "anger",
                "fear",
                "surprise",
                "disgust",
                "love",
                "excitement",
                "gratitude",
                "optimism",
                "pride",
                "disappointment",
                "anxiety",
                "confusion",
                "curiosity",
            ]

    def load_model(self, force=False):
        if self.model_loaded and not force:
            return

        self.tokenizer, self.model = self._load_llm_model()
        self.model_loaded = True

    def _load_llm_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )

        return tokenizer, model

    def predict(self, input_text: str) -> str:
        if not self.model_loaded:
            self.load_model()

        return self._get_llm_emotion_prediction(input_text, self.emotion_list)

    def predict_concept(self, input_text: str) -> str:
        return self._get_llm_emotion_prediction(input_text, self.emotion_list)

    def _get_llm_emotion_prediction(self, text: str, emotion_list: list) -> str:
        emotions_str = ", ".join(emotion_list)
        prompt = f"What emotion from this list: [{emotions_str}] best describes this text: '{text}' Answer with just the emotion name:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        response = generated_text.strip().lower()

        for emotion in emotion_list:
            if emotion.lower() in response:
                return emotion.lower()

        words = response.split()
        return words[0] if words else "unknown"

    def set_emotion_list(self, emotion_list: list):
        """Update the emotion list for predictions"""
        self.emotion_list = emotion_list

    def predict_with_custom_emotions(self, input_text: str, emotion_list: list) -> str:
        """Predict emotion with custom emotion list"""
        if not self.model_loaded:
            self.load_model()

        return self._get_llm_emotion_prediction(input_text, emotion_list)
