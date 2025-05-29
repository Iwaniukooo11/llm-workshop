# Abstract model interface

# Michał/Sebastian/Mateusz: Shared foundation
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from torch.utils.data import DataLoader

class BaseModel(ABC):
    """Abstract model interface (used by all team members)"""
    
    @abstractmethod
    def train(self, train_loader: DataLoader) -> None:
        """Train the model (implementation varies per contributor)"""
        pass
    
    @abstractmethod
    def predict(self, input_text: str) -> str:
        """Make prediction (implementation varies per contributor)"""
        pass