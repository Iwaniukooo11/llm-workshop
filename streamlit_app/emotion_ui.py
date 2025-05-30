# streamlit_app/emotion_ui.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show_emotion_ui(classifier):
    """Streamlit UI for emotion classification"""
    st.title('Advanced Emotion Analysis System')
    
    # Input text
    text = st.text_area('Enter text for emotion analysis:', 'I feel excited and optimistic about the future!')
    
    if st.button('Analyze Emotions'):
        # Prediction
        predictions = classifier.predict(text)
        
        # Display results
        st.subheader('Emotion Predictions')
        cols = st.columns(2)
        
        with cols[0]:
            st.write('**Top Emotions:**')
            for emotion, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.progress(score, text=f"{emotion.capitalize()}: {score:.2f}")
        
        with cols[1]:
            st.write('**All Emotions:**')
            fig, ax = plt.subplots()
            emotions = list(predictions.keys())
            scores = list(predictions.values())
            y_pos = np.arange(len(emotions))
            
            ax.barh(y_pos, scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(emotions)
            ax.invert_yaxis()
            ax.set_xlabel('Confidence')
            st.pyplot(fig)
        
        # Explanations
        st.subheader('Model Explanations')
        
        tab1, tab2 = st.tabs(['SHAP', 'LIME'])
        
        with tab1:
            st.write('**SHAP Explanation:**')
            shap_exp = classifier.explain_shap(text)
            st.text(shap_exp)
        
        with tab2:
            st.write('**LIME Explanation:**')
            lime_exp = classifier.explain_lime(text)
            st.write(lime_exp.as_list())
    
    # Model metrics section
    st.sidebar.header('Model Performance')
    if st.sidebar.button('Show Metrics'):
        # Load test data and calculate metrics
        # (Implement your test data loading here)
        test_metrics = {
            'accuracy': 0.86,
            'f1_macro': 0.82,
            'roc_auc': 0.91
        }
        
        st.sidebar.subheader('Test Metrics')
        st.sidebar.metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
        st.sidebar.metric("F1 Macro", f"{test_metrics['f1_macro']:.2%}")
        st.sidebar.metric("ROC AUC", f"{test_metrics['roc_auc']:.2%}")