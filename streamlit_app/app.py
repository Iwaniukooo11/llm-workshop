# Shared Component: Visualization (led by Mateusz)
import streamlit as st

def main():
    """Dashboard UI (Developed by Mateusz, uses all components)"""
    # Integrates Michał's models, Sebastian's experiments, and Mateusz's analysis
    st.title("LLM Interpretability Dashboard")
    
    # Uses all team members' components
    model = HuggingFaceModel()  # Michał's
    experiment = LLMSizeExperiment(model)  # Sebastian's
    interpreter = ModelInterpreter(model)  # Mateusz's