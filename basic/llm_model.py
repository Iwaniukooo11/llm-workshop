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
<<<<<<< HEAD
    def __init__(self, model_name="EleutherAI/gpt-j-6B"):
        super().__init__()
=======
    def __init__(
        self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", emotion_list=None, use_gpu=True
    ):
>>>>>>> b02edea90dcb230130236066f5e65a284b481f6e
        self.model_name = model_name
        self.tokenizer = None
<<<<<<< HEAD
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
=======
        self.use_gpu = use_gpu
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
>>>>>>> b02edea90dcb230130236066f5e65a284b481f6e

        self.training_data = []
        self.concept_description = ""

        self.prompt_header_llm_concept = "In 2 words guess, what task is the model doing, the format is x_test -> y_test:\n"
        self.prompt_content_llm_concept = "{x_test} -> {y_test}\n"
        self.prompt_tail_llm_concept = "What is this task?"

        self.prompt_header_llm_train = "You are a classificator\n"
        self.prompt_content_llm_train = "{x_train} -> {y_train}\n"
        self.prompt_tail_llm_train = "Learn based on this."

        self.prompt_llm_simulation = "{x_test}"
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

<<<<<<< HEAD
    def predict_concept(self, x_test, y_test):
        prompt = self.prompt_header_llm_concept
        for x, y in zip(x_test, y_test):
            prompt += self.prompt_content_llm_concept.format(x_test=x, y_test=y)
        prompt += self.prompt_tail_llm_concept
        print(f"Concept prediction prompt:\n{prompt}\n")
        concept = self._simple_prediction(prompt)
        self.concept_description = concept
        print(f"Predicted concept: {concept}")
        return concept

    def predict(self, x_test):
        if not self.training_data:
            raise ValueError(
                "Model must be trained first. Call train() with training data."
=======
    def _load_llm_model(self):
        if self.use_gpu:
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
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None
            )
            model.to(torch.device("cpu"))

        return tokenizer, model

    def predict(self, input_text: str) -> str:
        if not self.model_loaded:
            self.load_model()

        return self._get_llm_emotion_prediction(input_text, self.emotion_list)

    def predict_concept(self, input_text: str) -> str:
        return self._get_llm_emotion_prediction(input_text, self.emotion_list)

    def _get_llm_emotion_prediction(self, text: str, emotion_list: list) -> str:
        if not self.model_loaded:
            self.load_model()
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
>>>>>>> b02edea90dcb230130236066f5e65a284b481f6e
            )

        predictions = []

        prompt = self.prompt_header_llm_train

        for x_train, y_train in self.training_data:
            prompt += self.prompt_content_llm_train.format(
                x_train=x_train, y_train=y_train
            )

<<<<<<< HEAD
        prompt += self.prompt_tail_llm_train + "\n\n"
        prompt += f"Now predict the output for: {x_test}\n"
        prompt += "Output:"
        print(f"Prediction prompt for '{x_test}':\n{prompt}\n")
        prediction = self._simple_prediction(prompt)
        predictions.append(prediction)
        return predictions
=======
    def set_emotion_list(self, emotion_list: list):
        """Update the emotion list for predictions"""
        self.emotion_list = emotion_list

    def predict_with_custom_emotions(self, input_text: str, emotion_list: list) -> str:
        """Predict emotion with custom emotion list"""
        if not self.model_loaded:
            self.load_model()

        return self._get_llm_emotion_prediction(input_text, emotion_list)

    def train(self, train_loader: DataLoader) -> None:
        pass
>>>>>>> b02edea90dcb230130236066f5e65a284b481f6e
