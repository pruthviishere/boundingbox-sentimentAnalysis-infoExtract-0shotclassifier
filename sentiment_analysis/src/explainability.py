"""

# src/explainability.py
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients
 


def get_important_features_svm(vectorizer, svm_clf, class_names, top_n=20):
    """
    Extract important features from SVM model
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Trained TF-IDF vectorizer
    svm_clf : CalibratedClassifierCV
        Trained SVM classifier
    class_names : list
        Class names
    top_n : int
        Number of top features to extract
        
    Returns:
    --------
    dict
        Dictionary with important features for each class
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients
    if hasattr(svm_clf, 'coef_'):
        coefficients = svm_clf.coef_
    else:
        # For calibrated models, access the base estimator
        coefficients = svm_clf.base_estimator.coef_
    
    important_features = {}
    
    # Extract top features for each class
    for i, class_name in enumerate(class_names):
        # Get coefficients for class i
        class_coefficients = coefficients[i]
        
        # Sort indices by coefficient values
        sorted_indices = class_coefficients.argsort()
        
        # Get top positive features (most indicative of this class)
        top_positive_indices = sorted_indices[-top_n:]
        top_positive_features = [(feature_names[j], class_coefficients[j]) 
                                for j in top_positive_indices[::-1]]
        
        # Get top negative features (most contra-indicative of this class)
        top_negative_indices = sorted_indices[:top_n]
        top_negative_features = [(feature_names[j], class_coefficients[j]) 
                                for j in top_negative_indices]
        
        important_features[class_name] = {
            'positive': top_positive_features,
            'negative': top_negative_features
        }
    
    return important_features


def visualize_feature_importance(important_features, output_file=None):
    """
    Visualize important features for each class
    
    Parameters:
    -----------
    important_features : dict
        Dictionary with important features
    output_file : str, optional
        Output file to save visualization
    """
    num_classes = len(important_features)
    fig, axes = plt.subplots(num_classes, 2, figsize=(16, 5 * num_classes))
    
    # If there's only one class, wrap axes in a list for consistent indexing
    if num_classes == 1:
        axes = np.array([axes])
    
    for i, (class_name, features) in enumerate(important_features.items()):
        # Positive features
        pos_features = features['positive'][:15]  # Show top 15
        pos_words = [word for word, _ in pos_features]
        pos_scores = [score for _, score in pos_features]
        
        axes[i, 0].barh(pos_words[::-1], pos_scores[::-1], color='forestgreen')
        axes[i, 0].set_title(f'Top Positive Features for {class_name}')
        axes[i, 0].set_xlabel('Coefficient Value')
        
        # Negative features
        neg_features = features['negative'][:15]  # Show top 15
        neg_words = [word for word, _ in neg_features]
        neg_scores = [score for _, score in neg_features]
        
        axes[i, 1].barh(neg_words, neg_scores, color='crimson')
        axes[i, 1].set_title(f'Top Negative Features for {class_name}')
        axes[i, 1].set_xlabel('Coefficient Value')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Feature importance visualization saved to {output_file}")
    else:
        plt.show()


def explain_prediction_with_lime(text, vectorizer, model, class_names, num_features=10, 
                                show_prob=True, is_transformer=False):
    """
    Explain individual prediction using LIME
    
    Parameters:
    -----------
    text : str
        Text to explain
    vectorizer : TfidfVectorizer or None
        TF-IDF vectorizer for traditional model (None for transformer)
    model : object
        Trained model (SVM or Transformer)
    class_names : list
        Class names
    num_features : int
        Number of features to include in explanation
    show_prob : bool
        Whether to show prediction probabilities
    is_transformer : bool
        Whether the model is a transformer
        
    Returns:
    --------
    dict
        Explanation results
    """
    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Define prediction function based on model type
    if is_transformer:
        def predict_proba(texts):
            return model.predict_proba(texts)
    else:
        def predict_proba(texts):
            # Vectorize texts
            texts_tfidf = vectorizer.transform(texts)
            # Get probabilities
            return model.predict_proba(texts_tfidf)
    
    # Generate explanation
    exp = explainer.explain_instance(
        text, 
        predict_proba, 
        num_features=num_features
    )
    
    # Get prediction
    if is_transformer:
        pred_class = model.predict([text])[0]
        if show_prob:
            probs = model.predict_proba([text])[0]
    else:
        text_tfidf = vectorizer.transform([text])
        pred_class = model.predict(text_tfidf)[0]
        if show_prob:
            probs = model.predict_proba(text_tfidf)[0]
    
    # Extract explanation data
    explanation_data = {
        'text': text,
        'predicted_class': class_names[pred_class],
        'explanation': exp.as_list(),
    }
    
    if show_prob:
        explanation_data['probabilities'] = {
            class_name: float(probs[i]) for i, class_name in enumerate(class_names)
        }
    
    # Create visualization
    fig = plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for '{class_names[pred_class]}' prediction")
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    print("LIME explanation saved to 'lime_explanation.png'")
    
    return explanation_data


def visualize_transformer_attention(text, model, tokenizer, class_names, layer=-1, head=0):
    """
    Visualize attention weights from transformer model
    
    Parameters:
    -----------
    text : str
        Input text
    model : transformers.PreTrainedModel
        Trained transformer model
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer
    class_names : list
        Class names
    layer : int
        Attention layer to visualize
    head : int
        Attention head to visualize
        
    Returns:
    --------
    dict
        Attention visualization data
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get outputs with attention
    outputs = model(**inputs, output_attentions=True)
    
    # Get predicted class
    pred_class = outputs.logits.argmax(dim=1).item()
    
    # Get attention weights
    # attention shape: [batch, layers, heads, seq_len, seq_len]
    attention = outputs.attentions
    
    # Select specific layer and head
    if layer < 0:
        layer = len(attention) + layer  # Convert negative index
    
    selected_attention = attention[layer][0, head].detach().cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        selected_attention, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        cmap="YlOrRd"
    )
    plt.title(f"Attention (Layer {layer+1}, Head {head+1})\nPredicted: {class_names[pred_class]}")
    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    print("Attention visualization saved to 'attention_visualization.png'")
    
    # Return attention data
    attention_data = {
        'text': text,
        'predicted_class': class_names[pred_class],
        'tokens': tokens,
        'attention_weights': selected_attention.tolist()
    }
    
    return attention_data


def integrated_gradients_explain(text, model, tokenizer, class_names):
    """
    Explain prediction using Integrated Gradients
    
    Parameters:
    -----------
    text : str
        Input text
    model : transformers.PreTrainedModel
        Trained transformer model
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer
    class_names : list
        Class names
        
    Returns:
    --------
    dict
        Explanation data
    """
    # Tokenize
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predicted class
    model.eval()
    output = model(**inputs)
    pred_class = output.logits.argmax(dim=1).item()
    
    # Define prediction function
    def predict(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    # Initialize integrated gradients
    ig = LayerIntegratedGradients(
        predict, 
        model.bert.embeddings if hasattr(model, 'bert') else model.base_model.embeddings
    )
    
    # Get attribution
    attribution, delta = ig.attribute(
        inputs=inputs['input_ids'],
        baselines=None,  # Use all-zero embeddings as baseline
        additional_forward_args=(inputs['attention_mask'],),
        target=pred_class,
        n_steps=50,
        return_convergence_delta=True
    )
    
    # Extract word attributions
    attributions = attribution.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    
    # Get input tokens
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=attributions, 
        y=all_tokens,
        orient='h'
    )
    plt.title(f"Integrated Gradients Attribution for '{class_names[pred_class]}' prediction")
    plt.tight_layout()
    plt.savefig('integrated_gradients.png')
    print("Integrated Gradients visualization saved to 'integrated_gradients.png'")
    
    # Return attribution data
    attribution_data = {
        'text': text,
        'predicted_class': class_names[pred_class],
        'tokens': all_tokens,
        'attributions': attributions.tolist()
    }
    
    return attribution_data