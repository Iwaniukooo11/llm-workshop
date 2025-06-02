from core.base_model import BaseModel
from huggingface_hub import InferenceClient
import os
from typing import List, Any, Optional

class LLMModel(BaseModel):
    # Dictionary of supported models that are available on Hugging Face Inference API
    SUPPORTED_MODELS = {
        "llama-3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3-8b": "meta-llama/Llama-3.3-70B-Instruct",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gemma-7b": "google/gemma-7b-it"
    }
    
    def __init__(self, model_key="llama-3-8b", api_key='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'):
        super().__init__()
        
        # Use model key to get the actual model ID
        if model_key in self.SUPPORTED_MODELS:
            self.model_name = self.SUPPORTED_MODELS[model_key]
        else:
            self.model_name = model_key  # Allow custom model names
            
        # Set up API key, using environment variable as fallback
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API token not found. Set HF_API_KEY env variable or pass api_key.")
        
        self.client = None
        self.device = "api"  # Using API instead of local device

        self.training_data = []
        self.concept_description = ""

        self.prompt_header_llm_concept = "In 2 words guess, what task is the model doing, the format is x_test -> y_test:\n"
        self.prompt_content_llm_concept = "{x_test} -> {y_test}\n"
        self.prompt_tail_llm_concept = "What is this task?"

        self.prompt_header_llm_train = "You are a classificator\n"
        self.prompt_content_llm_train = "{x_train} -> {y_train}\n"
        self.prompt_tail_llm_train = "Learn based on this."

        self.prompt_llm_simulation = "{x_test}"
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the Hugging Face Inference Client"""
        try:
            print(f"Setting up Hugging Face client for model: {self.model_name}")
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=self.api_key
            )
            print(f"Client initialized successfully")
        except Exception as e:
            print(f"Error setting up client: {e}")
            raise

    def _simple_prediction(self, prompt, max_length=100, temperature=0.7):
        """Generate text using Hugging Face Inference API"""
        try:
            # Using the chat completions interface following the AgentHandler pattern
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature
            )
            
            # Extract the response text
            if hasattr(completion.choices[0], "message"):
                response = completion.choices[0].message.content
            else:
                response = str(completion.choices[0])
            
            # Remove the prompt if it appears at the beginning
            if prompt in response:
                response = response.replace(prompt, "").strip()
                
            return response if response else "No response generated"
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return f"Error generating prediction: {str(e)}"

    def train(self, x_train, y_train):
        """Store training examples and predict the concept"""
        self.training_data = []
        for x, y in zip(x_train, y_train):
            self.training_data.append((x, y))
        self.predict_concept(x_train, y_train)
        print(f"Training completed with {len(self.training_data)} examples")

    def predict_concept(self, x_test, y_test):
        """Predict the underlying concept from examples"""
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
        """Make predictions using stored training examples"""
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

        prompt += self.prompt_tail_llm_train + "\n\n"
        prompt += f"Now predict the output for: {x_test}\n"
        prompt += "Output (just plain answer):"
        print(f"Prediction prompt for '{x_test}':\n{prompt}\n")
        prediction = self._simple_prediction(prompt)
        predictions.append(prediction)
        return predictions