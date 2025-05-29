# Sebastian's Component: Intermediate experiments
from core.base_model import BaseModel
from typing import Dict, List

class LLMSizeExperiment:
    """LLM scaling experiments (Sebastian's responsibility)"""
    
    def __init__(self, model: BaseModel):
        self.model = model  # Uses others' model implementations
    
    def run(self, texts: List[str]) -> Dict[str, float]:
        # Sebastian's experiment logic here
        return {"gpt-3.5": 0.82, "gpt-4": 0.89}