# LLM model implementation
from torch.utils.data import DataLoader

from core.base_model import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch


class LLMModel(BaseModel):
    def __init__(self, model_name="EleutherAI/gpt-j-6B", verbose=False):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.training_data = []
        self.concept_description = ""

        self.prompt_header_llm_concept = "In 2 words guess, what task is the model doing, the format is x_test -> y_test:\n"
        self.prompt_content_llm_concept = "{x_test} -> {y_test}\n"
        self.prompt_tail_llm_concept = "What is this task?"

        self.prompt_header_llm_train = "You are a classificator\n"
        self.prompt_content_llm_train = "{x_train} -> {y_train}\n"
        self.prompt_tail_llm_train = "Learn based on this."

        self.prompt_llm_simulation = "{x_test}"
        self.verbose = verbose
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model.to(self.device)

            print(f"Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")

    def _simple_prediction(self, prompt, max_length=100, temperature=0.7):
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            return response if response else "No response generated"

        except Exception as e:
            print(f"Error in prediction: {e}")
            return

    def train(self, x_train, y_train):
        self.training_data = []
        for x, y in zip(x_train, y_train):
            self.training_data.append((x, y))
        self.predict_concept(x_train, y_train)
        print(f"Training completed with {len(self.training_data)} examples")

    def predict_concept(self, x_test, y_test):
        prompt = self.prompt_header_llm_concept
        for x, y in zip(x_test, y_test):
            prompt += self.prompt_content_llm_concept.format(x_test=x, y_test=y)
        prompt += self.prompt_tail_llm_concept
        if self.verbose:
            print(f"Concept prediction prompt:\n{prompt}\n")
        concept = self._simple_prediction(prompt, max_length=5, temperature=0.5)
        self.concept_description = concept
        if self.verbose:
            print(f"Predicted concept: {concept}")
        return concept

    def predict(self, x_test):
        if not self.training_data:
            raise ValueError(
                "Model must be trained first. Call train() with training data."
            )

        predictions = []

        prompt = self.prompt_header_llm_train

        for x_train, y_train in self.training_data:
            prompt += self.prompt_content_llm_train.format(
                x_train=x_train, y_train=y_train
            )

        prompt += self.prompt_tail_llm_train
        prompt += self.prompt_llm_simulation.format(x_test=x_test)
        if self.verbose:
            print(f"Prediction prompt for '{x_test}':\n{prompt}\n")
        prediction = self._simple_prediction(prompt, max_length=3, temperature=0.7)
        predictions.append(prediction.lower())
        return predictions[0]
