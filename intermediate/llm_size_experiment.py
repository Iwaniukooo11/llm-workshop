# Sebastian's Component: Intermediate experiments
from core.base_model import BaseModel
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.utils.data import DataLoader
import pandas as pd

# TMP FOR TESTING -----------------------------------------------
class HuggingFaceModel(BaseModel):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model = pipeline("text-classification", model=model_name)
    def predict(self, input_text: str) -> str:
        return self.model(input_text)[0]["label"]
    def train(self, train_loader: DataLoader) -> None:
        pass

class LLMClient:
    """
    Wrapper for a local HuggingFace causal LM.
    Supports generate() via `transformers` pipeline.
    """
    def __init__(self, model_name: str, device: Union[int, str] = 'cpu', **kwargs):
        if 'max_length' in kwargs:
            kwargs['max_new_tokens'] = kwargs.pop('max_length')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device != 'cpu':
            self.model.to(device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device!='cpu' else -1,
            **kwargs
        )
    def generate(self, prompt: str, system_message: str = None) -> str:
        full_prompt = (system_message + "\n\n" if system_message else "") + prompt
        outputs = self.generator(full_prompt, num_return_sequences=1)
        return outputs[0]["generated_text"][len(full_prompt):].strip()
# ---------------------------------------------------------------


# Experiments ----------------------

@dataclass
class PromptConfig:
    system:   Optional[str]  = None
    user_tmpl: str           = ""

class LLMSizeExperiment:

    def __init__(self):

        # Classifier outputs
        self.classifier_predicted_labels: Dict[str, List[str]] = {}
        self.classifier_accuracy: Dict[str, float] = {}

        # LLM concept guessing outputs and metrics
        self.llm_predicted_concepts: Dict[str, str] = {}
        self.llm_concept_accuracy: Dict[str, float] = {}

        # LLM simulation outputs and metrics
        self.llm_simulation_predicted_labels: Dict[str, List[str]] = {}
        self.llm_simulation_accuracy: Dict[str, float] = {}

        # LLM direct classification outputs and metrics
        self.llm_direct_predicted_labels: Dict[str, List[str]] = {}
        self.llm_direct_accuracy: Dict[str, float] = {}

    def run(
            self,
            *,
            X: List[str],
            y: List[str],
            dataset_name: str,
            shots: int,
            concept: str,
            concept_keywords: List[str],
            classifier_name: str,
            classifier: BaseModel,
            llm_clients: Dict[str, "LLMClient"],
            prompt_llm_concept: PromptConfig,
            prompt_llm_simulation: PromptConfig,
            prompt_llm_direct: PromptConfig
    ) -> Tuple['pd.DataFrame', 'pd.DataFrame']:
        """
        Runs a battery of experiments to compare classifier behavior and LLM interpretability at scale.

        This method conducts four main tasks:
        1. **Classifier accuracy**: runs the classifier on the entire dataset and measures y_pred vs y.
        2. **Concept Guessing**: shows the LLM a few (X→y_pred) examples and asks it to name the task.
        3. **Simulation**: asks the LLM to replicate the classifier's predictions on held-out inputs.
        4. **Direct Classification**: asks the LLM to label inputs directly, measuring against ground truth y.

        Each task is run for every model in `llm_clients`. The results are returned as two pandas DataFrames:
        - **metrics_df**: one row per (dataset, classifier, llm) with accuracy metrics for all four tasks
        - **preds_df**: one row per (dataset, classifier, llm, input) with raw predictions for downstream analysis

        Parameters
        ----------
        X : List[str] - Text inputs for experiments.
        y : List[str] - Ground truth labels corresponding to X.
        dataset_name : str - Identifier for the dataset (used in DataFrame outputs).
        shots : int - Number of few-shot examples to prime the LLM.
        concept : str - The true task name, used to evaluate concept guessing.
        concept_keywords : List[str] - Keywords to match in the LLM's concept-guess output.
        classifier_name : str - A short label for the classifier (e.g., 'distilbert-sst2').
        classifier : BaseModel - An instance implementing the BaseModel interface (train/predict).
        llm_clients : Dict[str, LLMClient] - Mapping from LLM identifiers to LLMClient instances.
        prompt_llm_concept : PromptConfig- Template and optional system message for concept-guess prompts.
        prompt_llm_simulation : PromptConfig- Template and optional system message for simulation prompts.
        prompt_llm_direct : PromptConfig - Template and optional system message for direct classification prompts.

        Returns
        -------
        metrics_df : pd.DataFrame
        preds_df : pd.DataFrame
        """

        # 1) classifier prediction
        classifier_predicted_labels = [classifier.predict(text) for text in X]
        self.classifier_predicted_labels[classifier_name] = classifier_predicted_labels
        self.classifier_accuracy[classifier_name] = (
            sum(pred == true for pred, true in zip(classifier_predicted_labels, y))/ len(y)
        )

        # 2) for each LLM, do concept‐guess / simulation prediction / direct prediction
        context_for_llm = list(zip(X[:shots], y[:shots]))
        X_for_llm = X[shots:]
        y_for_llm = y[shots:]
        for name, client in llm_clients.items():
            print('Running experiment for LLM:', name)

            shot_txt = "\n".join(f"{i + 1}. \"{t}\" → {lab}" for i, (t, lab) in enumerate(context_for_llm))

            # --- Concept Guess ---
            system_prompt_llm_concept = prompt_llm_concept.system
            user_prompt_llm_concept = prompt_llm_concept.user_tmpl.format(shot=shot_txt)
            llm_concept_prediction = client.generate(
                user_prompt_llm_concept,
                system_message=system_prompt_llm_concept).strip()
            self.llm_predicted_concepts[name] = llm_concept_prediction
            self.llm_concept_accuracy[name] = float(concept.lower() in llm_concept_prediction.lower()
                                                    or any(kw.lower() in llm_concept_prediction.lower()
                                                           for kw in concept_keywords))

            # --- Simulation ---
            llm_simulation_predicted_labels: List[str] = []
            system_prompt_llm_simulation = prompt_llm_simulation.system
            for text in X_for_llm:
                user_prompt_llm_simulation = prompt_llm_simulation.user_tmpl.format(shot=shot_txt, input=text)
                llm_simulation_predicted_label = client.generate(
                    user_prompt_llm_simulation,
                    system_message=system_prompt_llm_simulation).split()[0].strip()
                llm_simulation_predicted_labels.append(llm_simulation_predicted_label)
            self.llm_simulation_predicted_labels[name] = llm_simulation_predicted_labels
            self.llm_simulation_accuracy[name] = sum(simulation_label == classifier_label
                                                     for simulation_label, classifier_label in zip(llm_simulation_predicted_labels,
                                                     classifier_predicted_labels[shots:])) / len(llm_simulation_predicted_labels)

            # --- Direct Performance ---
            llm_direct_predicted_labels: List[str] = []
            system_prompt_llm_direct = prompt_llm_direct.system
            for text in X_for_llm:
                user_prompt_llm_direct = prompt_llm_direct.system.format(input=text)
                llm_direct_predicted_label = client.generate(
                    user_prompt_llm_direct,
                    system_message=system_prompt_llm_direct).split()[0].strip()
                llm_direct_predicted_labels.append(llm_direct_predicted_label)
            self.llm_direct_predicted_labels[name] = llm_direct_predicted_labels
            self.llm_direct_accuracy[name] = sum(l == t
                                                 for l, t in zip(llm_direct_predicted_labels, y_for_llm)) / len(llm_direct_predicted_labels)

            # 4) build metrics DataFrame
            metrics_rows = []
            for clf in self.classifier_accuracy:
                for llm in self.llm_predicted_concepts:
                    metrics_rows.append({
                        'dataset_name': dataset_name,
                        'classifier': clf,
                        'llm': llm,
                        'classifier_accuracy': self.classifier_accuracy[clf],
                        'llm_concept_accuracy': self.llm_concept_accuracy.get(llm, 0.0),
                        'llm_simulation_accuracy': self.llm_simulation_accuracy.get(llm, 0.0),
                        'llm_direct_accuracy': self.llm_direct_accuracy.get(llm, 0.0)
                    })
            metrics_df = pd.DataFrame(metrics_rows)

            # 5) build predictions DataFrame
            preds_rows = []
            for clf in self.classifier_predicted_labels:
                for llm in self.llm_predicted_concepts:
                    concept_guess = self.llm_predicted_concepts[llm]
                    sim_list = self.llm_simulation_predicted_labels[llm]
                    dir_list = self.llm_direct_predicted_labels[llm]
                    for idx, text in enumerate(X):
                        row = {
                            'dataset_name': dataset_name,
                            'classifier_name': clf,
                            'llm_name': llm,
                            'X': text,
                            'y': y[idx],
                            'classifier_predicted_label': self.classifier_predicted_labels[clf][idx],
                            'llm_predicted_concept': concept_guess if idx == 0 else None,
                            'llm_simulation_predicted_label': (sim_list[idx - shots] if idx >= shots else None),
                            'llm_direct_predicted_label': (dir_list[idx - shots] if idx >= shots else None)
                        }
                        preds_rows.append(row)
            predictions_df = pd.DataFrame(preds_rows)

        return metrics_df, predictions_df
    

if __name__ == "__main__":

    data = {
        "text": [
            "I absolutely loved this movie, it was fantastic!",
            "Worst experience ever. I hated every second.",
            "The plot was not intriguing and the characters were flat.",
            "Fantastic performance by the lead actor!",
            "I wouldn't recommend this to anyone.",
            "It was not an okay film, boring.",
            "What a masterpiece! Truly a work of art.",
            "It bored me to tears, very dull.",
        ],
        "label": [
            "POSITIVE",
            "NEGATIVE",
            "NEGATIVE",
            "POSITIVE",
            "NEGATIVE",
            "NEGATIVE",
            "POSITIVE",
            "NEGATIVE",
        ]
    }

    experiment = LLMSizeExperiment()
    metrics_df, precictions_df = experiment.run(
        X=data["text"],
        y=data["label"],
        dataset_name='test_dataset',
        shots=4,
        concept="sentiment analysis",
        concept_keywords=["sentiment", "emotion"],

        classifier_name="distilbert-sst2",
        classifier=HuggingFaceModel("distilbert-base-uncased-finetuned-sst-2-english"),

        llm_clients={
            "gpt2-124M": LLMClient("gpt2", max_length=5, temperature=1.0),
            "gpt2-345M": LLMClient("gpt2-medium", max_length=5, temperature=1.0),
            "gpt2-774M": LLMClient("gpt2-large", max_length=5, temperature=1.0)
        },
        prompt_llm_concept=PromptConfig(
            system   = "In 2 words guess, what task is the model doing:",
            user_tmpl= "{shot}\n\nBased on these examples, what task is the model performing?"
        ),
        prompt_llm_simulation=PromptConfig(
            system   = "You are a classifier. Reply with exactly one word: POSITIVE or NEGATIVE.",
            user_tmpl= "{shot}\n\nLabel this: \"{input}\""
        ),
        prompt_llm_direct=PromptConfig(
            system   = "You are a classifier. Reply with exactly one word: POSITIVE or NEGATIVE.",
            user_tmpl= "\"{input}\":"
        )
    )
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(metrics_df)
    print(precictions_df)


