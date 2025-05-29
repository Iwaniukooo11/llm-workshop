import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import shap


class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels_indices: List[int], tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels_indices = labels_indices 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        text = str(self.texts[item])
        label_idx = self.labels_indices[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_idx, dtype=torch.long) 
        }

class EmotionBERT(nn.Module):
    """Custom BERT-based emotion classifier for single-label classification."""
    def __init__(self, n_classes: int):
        super(EmotionBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits 

class EmotionClassifier:
    """Complete system for single-label emotion classification."""
    def __init__(self, model_name: str = 'bert-base-uncased', device: Optional[str] = None,
                 max_len: int = 128):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model: Optional[EmotionBERT] = None
        self.max_len = max_len
        
        self.label_to_idx: Optional[Dict[Any, int]] = None
        self.idx_to_label: Optional[Dict[int, Any]] = None
        self.num_labels: Optional[int] = None
        self.emotion_labels_ordered: Optional[List[str]] = None

        
        
        


    def _setup_labels(self, train_labels_raw: List[Any]):
        """
        Determines unique labels from raw training labels, creates mappings,
        and sets the number of labels and their ordered list.
        """
        if not train_labels_raw:
            raise ValueError("Training labels cannot be empty for label setup.")
            
        unique_labels = sorted(list(set(train_labels_raw)))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.num_labels = len(unique_labels)
        self.emotion_labels_ordered = [str(self.idx_to_label[i]) for i in range(self.num_labels)] 
        
        print(f"Labels setup: {self.num_labels} unique labels found: {self.emotion_labels_ordered}")


    def train(self, train_texts: List[str], train_labels_raw: List[Any],
              val_texts: List[str], val_labels_raw: List[Any],
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Training pipeline with validation for single-label classification."""

        self._setup_labels(train_labels_raw) 
        
        if self.num_labels is None or self.label_to_idx is None:
            raise RuntimeError("Label setup failed. num_labels or label_to_idx is not set.")

        self.model = EmotionBERT(n_classes=self.num_labels).to(self.device)

        
        try:
            train_labels_idx = [self.label_to_idx[lbl] for lbl in train_labels_raw]
            val_labels_idx = [self.label_to_idx[lbl] for lbl in val_labels_raw]
        except KeyError as e:
            raise ValueError(f"Label '{e.args[0]}' in validation data not found in training data labels. Ensure all validation labels are present in training labels.")


        train_dataset = EmotionDataset(train_texts, train_labels_idx, self.tokenizer, self.max_len)
        val_dataset = EmotionDataset(val_texts, val_labels_idx, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss() 

        best_f1 = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                print(f"Processing batch {batch_idx + 1}/{len(train_loader)}", end='\r')
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device) 

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'  Val F1 Macro: {val_metrics["f1_macro"]:.4f}')
            print(f'  Val ROC AUC (Macro OVR): {val_metrics["roc_auc"]:.4f}')

            current_f1 = val_metrics.get("f1_macro", 0.0)
            if current_f1 >= best_f1: 
                best_f1 = current_f1
                torch.save(self.model.state_dict(), 'best_model_state.pth')
                print(f"  New best model saved with F1 Macro: {best_f1:.4f}")
        
        print("Training finished. Loading best model state.")
        if self.model:
            self.model.load_state_dict(torch.load('best_model_state.pth'))
        else:
            
            print("Error: Model was not initialized before attempting to load state dict.")


    def predict(self, text: str) -> Dict[str, float]:
        """Predicts the single most probable emotion for a given text."""
        if not self.model or not self.idx_to_label or not self.emotion_labels_ordered:
            raise RuntimeError("Model is not trained or label mappings are not set. Call train() first.")
        
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        
        predicted_idx = np.argmax(probabilities)
        predicted_label_name = self.emotion_labels_ordered[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return {predicted_label_name: float(confidence)}

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Calculates evaluation metrics for single-label classification."""
        if not self.model or not self.num_labels or not self.emotion_labels_ordered:
            raise RuntimeError("Model is not trained or essential attributes (num_labels, emotion_labels_ordered) are not set.")

        self.model.eval()
        all_predicted_probs = []
        all_true_indices = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                true_labels_batch = batch['labels'].cpu().numpy() 
                
                logits = self.model(input_ids, attention_mask)
                probs_batch = torch.softmax(logits, dim=1).cpu().numpy() 
                
                all_predicted_probs.append(probs_batch)
                all_true_indices.append(true_labels_batch)

        all_predicted_probs = np.concatenate(all_predicted_probs, axis=0) 
        all_true_indices = np.concatenate(all_true_indices, axis=0)       

        predicted_indices = np.argmax(all_predicted_probs, axis=1) 

        accuracy = accuracy_score(all_true_indices, predicted_indices)
        f1 = f1_score(all_true_indices, predicted_indices, average='macro', zero_division=0)
        
        roc_auc = 0.0
        if self.num_labels > 1: 
            try:
                
                
                roc_auc = roc_auc_score(all_true_indices, all_predicted_probs, average='macro', multi_class='ovr')
            except ValueError as e:
                print(f"Warning: Could not compute ROC AUC. Error: {e}. Setting ROC AUC to 0.")
                roc_auc = 0.0
        else: 
            print("Warning: Only one class detected or specified; ROC AUC is not computed.")


        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'roc_auc': roc_auc
        }

    def explain_shap(self, text: str) -> shap.Explanation:
        """Generates SHAP explanation for a single text input."""
        if not self.model or not self.emotion_labels_ordered:
            raise RuntimeError("Model is not trained or emotion_labels_ordered is not set. Call train() first.")

        self.model.eval()
        
        def predictor_for_shap(texts: List[str]): 
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = torch.softmax(logits, dim=1) 
            return probabilities.cpu().numpy() 

        explainer = shap.Explainer(predictor_for_shap, self.tokenizer, output_names=self.emotion_labels_ordered)
        shap_values = explainer([text]) 
        return shap_values

    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    

    
    
    


if __name__ == '__main__':
    
    sample_train_texts = [
        "I am so happy today!", "This is really sad news.", "I feel quite angry about this.",
        "Feeling joyful and excited.", "What a depressing situation.", "He was furious with the outcome."
    ]
    
    sample_train_labels_raw = ["happy", "sad", "angry", "happy", "sad", "angry"]

    sample_val_texts = ["This is great!", "I am very upset."]
    sample_val_labels_raw = ["happy", "angry"] 

    print("Initializing EmotionClassifier...")
    classifier = EmotionClassifier(device='cpu') 

    print("\nStarting training...")
    classifier.train(
        sample_train_texts, sample_train_labels_raw,
        sample_val_texts, sample_val_labels_raw,
        epochs=2, batch_size=2 
    )

    print("\n--- Training Complete ---")

    test_text_1 = "I'm thrilled!"
    prediction_1 = classifier.predict(test_text_1)
    print(f"\nPrediction for '{test_text_1}': {prediction_1}")

    test_text_2 = "This makes me so mad."
    prediction_2 = classifier.predict(test_text_2)
    print(f"Prediction for '{test_text_2}': {prediction_2}")

    print("\nGenerating SHAP explanation for the first test text...")
    try:
        shap_explanation = classifier.explain_shap(test_text_1)
        
        
        
        print(f"SHAP values object type: {type(shap_explanation)}")
        print(f"SHAP base values: {shap_explanation.base_values}")
        

    except Exception as e:
        print(f"Error generating SHAP explanation: {e}")
    
    
    
    
    
    
    
    

    print("\n--- Script Finished ---")