from pandas import DataFrame
from core.base_model import BaseModel
from typing import Dict, List
from collections import Counter
import pandas as pd

class LLMSizeExperiment:

    def __init__(self):

        self.run_number: int = 0

        # Classifier outputs
        self.classifier_predicted_labels: Dict[str, List[str]] = {}
        self.classifier_accuracy: Dict[str, float] = {}
        self.classifier_confidence: Dict[str, List[float]] = {}
        # LLM concept guessing outputs and metrics
        self.llm_predicted_concepts: Dict[str, str] = {}
        self.llm_concept_accuracy: Dict[str, float] = {}
        # LLM simulation outputs and metrics
        self.llm_simulation_predicted_labels: Dict[str, List[str]] = {}
        self.llm_simulation_accuracy: Dict[str, float] = {}
        self.llm_direct_prediction_accuracy: Dict[str, float] = {}

    def run(
            self,
            *,
            x_train: List[str],
            y_train: List[str],
            x_test: List[str],
            y_test: List[str],
            x_val: List[str],
            y_val: List[str],
            dataset_name: str,
            concept: str,
            concept_keywords: List[str],

            classifier_name: str,
            classifier: BaseModel,
            train_classifier: bool,
            classifier_train_arguments: Dict[str, int],

            llm_models: Dict[str, BaseModel],
            prompt_header_llm_concept: str,
            prompt_content_llm_concept: str,
            prompt_tail_llm_concept: str,
            prompt_header_llm_train: str,
            prompt_content_llm_train: str,
            prompt_tail_llm_train: str,
            prompt_llm_simulation: str,
    ) -> tuple[DataFrame, DataFrame, DataFrame]:

        self.run_number += 1

        # 0) clear previous data
        # Classifier outputs
        self.classifier_predicted_labels: Dict[str, List[str]] = {}
        self.classifier_accuracy: Dict[str, float] = {}
        self.classifier_confidence: Dict[str, List[float]] = {}
        # LLM concept guessing outputs and metrics
        self.llm_predicted_concepts: Dict[str, str] = {}
        self.llm_concept_accuracy: Dict[str, float] = {}
        # LLM simulation outputs and metrics
        self.llm_simulation_predicted_labels: Dict[str, List[str]] = {}
        self.llm_simulation_accuracy: Dict[str, float] = {}
        self.llm_direct_prediction_accuracy: Dict[str, float] = {}

        # 1) classifier prediction
        if train_classifier:
            classifier.train(x_train, y_train, x_val, y_val, **classifier_train_arguments)
        raw_predictions = [classifier.predict(text) for text in x_test]
        classifier_predicted_labels = [list(d.keys())[0] for d in raw_predictions]
        classifier_predicted_confidences = [list(d.values())[0] for d in raw_predictions]
        self.classifier_predicted_labels[classifier_name] = classifier_predicted_labels
        self.classifier_confidence[classifier_name] = classifier_predicted_confidences
        self.classifier_accuracy[classifier_name] = (
            sum(pred == y for pred, y in zip(classifier_predicted_labels, y_test))/ len(y_test)
        )

        # 2) for each LLM, do concept‐guess / training / simulation prediction
        for name, llm_model in llm_models.items():
            print('Running experiment for LLM:', name)

            # --- Concept Guess ---
            prompt_content = ""
            for x, y in zip(x_test, y_test):
                prompt_content += prompt_content_llm_concept.format(x_test=x, y_test=y)
            prompt = prompt_header_llm_concept + prompt_content + prompt_tail_llm_concept
            llm_concept_prediction = "Predicted concept" #llm_model.predict_concept(input_text=prompt) TODO
            self.llm_predicted_concepts[name] = llm_concept_prediction
            self.llm_concept_accuracy[name] = float(concept.lower() in llm_concept_prediction.lower()
                                                    or any(kw.lower() in llm_concept_prediction.lower()
                                                           for kw in concept_keywords))

            # --- Training ---
            prompt_content = ""
            for x, y in zip(x_train, y_train):
                prompt_content += prompt_content_llm_train.format(x_train=x, y_train=y)
            prompt = prompt_header_llm_train + prompt_content + prompt_tail_llm_train
            llm_model.train(train_loader=None, x_train=x_train, y_train=y_train, prompt=prompt)

            # --- Simulation ---
            llm_simulation_predicted_labels: List[str] = []
            for x in x_test:
                prompt = prompt_llm_simulation.format(x_test=x)
                llm_simulation_predicted_label = llm_model.predict(input_text=prompt)
                llm_simulation_predicted_labels.append(llm_simulation_predicted_label)
            self.llm_simulation_predicted_labels[name] = llm_simulation_predicted_labels
            self.llm_simulation_accuracy[name] = sum(simulation_label == classifier_label
                                                     for simulation_label, classifier_label in zip(llm_simulation_predicted_labels,
                                                     classifier_predicted_labels)) / len(llm_simulation_predicted_labels)
            self.llm_direct_prediction_accuracy[name] = sum(simulation_label == true_label
                                                     for simulation_label, true_label in zip(llm_simulation_predicted_labels,
                                                    y_test)) / len(llm_simulation_predicted_labels)

        # 4) build model DataFrame
        rows = []
        for llm_name, _ in llm_models.items():
            rows.append({
                'run_id': self.run_number,
                'dataset_name': dataset_name,
                'classifier': classifier_name,
                'llm': llm_name,
                'classifier_accuracy': self.classifier_accuracy[classifier_name],
                'llm_concept_accuracy': self.llm_concept_accuracy.get(llm_name, 0.0),
                'llm_simulation_accuracy': self.llm_simulation_accuracy.get(llm_name, 0.0),
                'llm_direct_prediction_accuracy': self.llm_direct_prediction_accuracy.get(llm_name, 0.0),
                'prompt_header_llm_concept': prompt_header_llm_concept,
                'prompt_content_llm_concept': prompt_content_llm_concept,
                'prompt_tail_llm_concept': prompt_tail_llm_concept,
                'prompt_header_llm_train': prompt_header_llm_train,
                'prompt_content_llm_train': prompt_content_llm_train,
                'prompt_tail_llm_train': prompt_tail_llm_train,
                'prompt_llm_simulation': prompt_llm_simulation,
                'llm_predicted_concept': self.llm_predicted_concepts.get(llm_name, 0.0),
            })
        model_statistics = pd.DataFrame(rows)

        # 5) build predictions DataFrame
        rows = []
        for llm_name, _ in llm_models.items():
            for (x, y,
                 classifier_predicted_label,
                 llm_simulation_predicted_label,
                 classifier_predicted_label_confidence) in zip(x_test, y_test,
                                                        self.classifier_predicted_labels[classifier_name],
                                                        self.llm_simulation_predicted_labels[llm_name],
                                                        self.classifier_confidence[classifier_name]):
                rows.append({
                    'run_id': self.run_number,
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'llm_name': llm_name,
                    'x_test': x,
                    'y_test': y,
                    'classifier_predicted_label': classifier_predicted_label,
                    'classifier_predicted_label_confidence': classifier_predicted_label_confidence,
                    'llm_simulation_predicted_label': llm_simulation_predicted_label
                })
        prediction_statistics = pd.DataFrame(rows)

        # 6) build data DataFrame
        rows = []
        for x_data, y_data, name_data in zip([x_train, x_val, x_test], [y_train, y_val, y_test], ['train', 'val', 'test']):
            counts = Counter(y_data)
            train_props = {label: count / len(y_data)
                           for label, count in counts.items()}
            avg_text_length = sum(len(txt) for txt in x_data) / len(x_data)
            avg_word_count = sum(len(txt.split()) for txt in x_data) / len(x_data)
            rows.append({
                'run_id': self.run_number,
                'dataset_name': dataset_name,
                'classifier_name': classifier_name,
                'partition': name_data,
                'num_samples': len(x_data),
                'label_counts': str(dict(counts)),
                'label_proportions': str({k: round(v, 2) for k, v in train_props.items()}),
                'avg_text_length': round(avg_text_length, 1),
                'avg_word_count': round(avg_word_count, 1)
            })
        data_statistics = pd.DataFrame(rows)

        return model_statistics, prediction_statistics, data_statistics