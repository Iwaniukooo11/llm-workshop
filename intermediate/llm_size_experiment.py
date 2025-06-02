from pandas import DataFrame
from core.base_model import BaseModel
from typing import Dict, List
from collections import Counter
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)

class LLMSizeExperiment:

    def __init__(self):

        # Run counter
        self.run_number: int = 0
        # Classifier outputs
        self.classifier_predicted_labels: Dict[str, List[str]] = {}
        self.classifier_accuracy: Dict[str, float] = {}
        self.classifier_precision: Dict[str, float] = {}
        self.classifier_recall: Dict[str, float] = {}
        self.classifier_f1: Dict[str, float] = {}
        self.classifier_balanced_accuracy: Dict[str, float] = {}
        self.classifier_cohen_kappa: Dict[str, float] = {}
        self.classifier_mcc: Dict[str, float] = {}
        self.classifier_confidence: Dict[str, List[float]] = {}
        # LLM concept guessing outputs and metrics
        self.llm_predicted_concepts: Dict[str, str] = {}
        # LLM simulation outputs and metrics
        self.llm_simulation_predicted_labels: Dict[str, List[str]] = {}
        self.simulation_correct: Dict[str, List[bool]] = {}

    def run(
            self,
            *,
            x_train: List[str],
            y_train: List[str],
            x_test: List[str],
            y_test: List[str],
            x_val: List[str],
            y_val: List[str],
            max_samples_for_llm_train: int,
            dataset_name: str,
            concept: str,
            concept_keywords: List[str],
            max_samples_for_concept: int,

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
        self.classifier_precision: Dict[str, float] = {}
        self.classifier_recall: Dict[str, float] = {}
        self.classifier_f1: Dict[str, float] = {}
        self.classifier_balanced_accuracy: Dict[str, float] = {}
        self.classifier_cohen_kappa: Dict[str, float] = {}
        self.classifier_mcc: Dict[str, float] = {}
        self.classifier_confidence: Dict[str, List[float]] = {}
        # LLM concept guessing outputs and metrics
        self.llm_predicted_concepts: Dict[str, str] = {}
        # LLM simulation outputs and metrics
        self.llm_simulation_predicted_labels: Dict[str, List[str]] = {}
        self.simulation_correct: Dict[str, List[bool]] = {}

        # 1) classifier prediction
        if train_classifier:
            print('Training classifier...')
            classifier.train(x_train, y_train, x_val, y_val, **classifier_train_arguments)
        print('Classifier is predicting sentiments...')
        raw_predictions = [classifier.predict(text) for text in x_test]
        classifier_predicted_labels = [list(d.keys())[0] for d in raw_predictions]
        classifier_predicted_confidences = [list(d.values())[0] for d in raw_predictions]
        # Store classifier outputs
        self.classifier_predicted_labels[classifier_name] = classifier_predicted_labels
        self.classifier_confidence[classifier_name] = classifier_predicted_confidences
        # Compute classifier metrics:
        cls_acc = accuracy_score(y_test, classifier_predicted_labels)
        cls_prec = precision_score(y_test, classifier_predicted_labels, average="macro", zero_division=0)
        cls_rec = recall_score(y_test, classifier_predicted_labels, average="macro", zero_division=0)
        cls_f1 = f1_score(y_test, classifier_predicted_labels, average="macro", zero_division=0)
        cls_bal_acc = balanced_accuracy_score(y_test, classifier_predicted_labels)
        cls_kappa = cohen_kappa_score(y_test, classifier_predicted_labels)
        try:
            cls_mcc = matthews_corrcoef(y_test, classifier_predicted_labels)
        except Exception:
            cls_mcc = 0.0
        self.classifier_accuracy[classifier_name] = cls_acc
        self.classifier_precision[classifier_name] = cls_prec
        self.classifier_recall[classifier_name] = cls_rec
        self.classifier_f1[classifier_name] = cls_f1
        self.classifier_balanced_accuracy[classifier_name] = cls_bal_acc
        self.classifier_cohen_kappa[classifier_name] = cls_kappa
        self.classifier_mcc[classifier_name] = cls_mcc
        print('Classifying task done.\n')

        # 2) for each LLM, do concept‐guess / training / simulation prediction
        for name, llm_model in llm_models.items():
            print('Running experiment for LLM:', name, '-----------------')

            # --- Concept Guess ---
            print('LLM guessing context...', end='')
            llm_model.prompt_header_llm_concept = prompt_header_llm_concept
            llm_model.prompt_content_llm_concept = prompt_content_llm_concept
            llm_model.prompt_tail_llm_concept = prompt_tail_llm_concept
            llm_concept_prediction = llm_model.predict_concept(x_test[:max_samples_for_concept], classifier_predicted_labels[:max_samples_for_concept])
            self.llm_predicted_concepts[name] = llm_concept_prediction
            print('guessed: ', llm_concept_prediction)

            # --- Training ---
            print("Training LLM based on classifier's inputs and outputs...")
            llm_model.prompt_header_llm_train = prompt_header_llm_train
            llm_model.prompt_header_llm_train = prompt_content_llm_train
            llm_model.prompt_header_llm_train = prompt_tail_llm_train
            llm_model.train(x_test[:max_samples_for_llm_train], classifier_predicted_labels[:max_samples_for_llm_train])

            # --- Simulation ---
            print("LLM simulating classifier...")
            llm_simulation_predicted_labels: List[str] = []
            llm_model.prompt_llm_simulation = prompt_llm_simulation
            for i, x in enumerate(x_test):
                llm_simulation_predicted_label = llm_model.predict(x)
                print('true label:', y_test[i], ' | classifier label:',
                      self.classifier_predicted_labels[classifier_name][i],
                      ' | LLM label:', llm_simulation_predicted_label)
                llm_simulation_predicted_labels.append(llm_simulation_predicted_label)
            self.llm_simulation_predicted_labels[name] = llm_simulation_predicted_labels
            self.simulation_correct[name] = [classifier_label.lower() in sim_label.lower()
                                             for sim_label, classifier_label in zip(
                    llm_simulation_predicted_labels, classifier_predicted_labels)]
            print('Simulation done.\n')

        # 4) build model DataFrame
        rows = []
        for llm_name, _ in llm_models.items():
            rows.append({
                'run_id': self.run_number,
                'dataset_name': dataset_name,
                'classifier': classifier_name,
                'llm': llm_name,
                # Classifier metrics
                'classifier_accuracy': self.classifier_accuracy[classifier_name],
                'classifier_precision': self.classifier_precision[classifier_name],
                'classifier_recall': self.classifier_recall[classifier_name],
                'classifier_f1': self.classifier_f1[classifier_name],
                'classifier_balanced_accuracy': self.classifier_balanced_accuracy[classifier_name],
                'classifier_cohen_kappa': self.classifier_cohen_kappa[classifier_name],
                'classifier_mcc': self.classifier_mcc[classifier_name],
                # Prompts
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
                 x_test_present_in_prompt,
                 classifier_predicted_label_confidence,
                 simulation_correct                     ) in zip(x_test, y_test,
                                                        self.classifier_predicted_labels[classifier_name],
                                                        self.llm_simulation_predicted_labels[llm_name],
                                                        [True] * max_samples_for_llm_train + [False] * (len(self.llm_simulation_predicted_labels[llm_name]) - max_samples_for_llm_train),
                                                        self.classifier_confidence[classifier_name],
                                                        self.simulation_correct[llm_name]):
                rows.append({
                    'run_id': self.run_number,
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'llm_name': llm_name,
                    'x_test': x,
                    'y_test': y,
                    'classifier_predicted_label': classifier_predicted_label,
                    'classifier_predicted_label_confidence': classifier_predicted_label_confidence,
                    'x_test_present_in_prompt': x_test_present_in_prompt,
                    'llm_simulation_label_correct': simulation_correct,
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