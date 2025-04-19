# src/evaluation.py
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_and_compare(
    X_test, y_test, 
    tfidf_svm_model, transformer_model, 
    class_names=['Negative',   'Positive']
):
    """
    Evaluate and compare both models
    
    Parameters:
    -----------
    X_test : array-like
        Test text data
    y_test : array-like
        Test labels
    tfidf_svm_model : TfidfSvmModel
        Trained TF-IDF + SVM model
    transformer_model : TransformerModel
        Trained transformer model
    class_names : list
        Class names for visualization
        
    Returns:
    --------
    dict
        Evaluation results
    """
    print("Evaluating models on test data...")
    
    # Predictions
    y_pred_svm = tfidf_svm_model.predict(X_test)
    y_pred_transformer = transformer_model.predict(X_test)
    
    # Get reports
    svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
    transformer_report = classification_report(y_test, y_pred_transformer, output_dict=True)
    
    # Compare results
    comparison = pd.DataFrame({
        'TF-IDF + SVM': [
            svm_report['weighted avg']['precision'], 
            svm_report['weighted avg']['recall'],
            svm_report['weighted avg']['f1-score'],
            svm_report['accuracy'],
            tfidf_svm_model.training_time,
            tfidf_svm_model.inference_time * 1000  # Convert to ms
        ],
        'Transformer': [
            transformer_report['weighted avg']['precision'], 
            transformer_report['weighted avg']['recall'],
            transformer_report['weighted avg']['f1-score'],
            transformer_report['accuracy'],
            transformer_model.training_time,
            transformer_model.inference_time * 1000  # Convert to ms
        ]
    }, index=['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Training Time (s)', 'Inference Time (ms)'])
    
    print("\nModel Comparison:")
    print(comparison)
    
    # Calculate confusion matrices
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_transformer = confusion_matrix(y_test, y_pred_transformer)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # SVM confusion matrix
    sns.heatmap(
        cm_svm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names, 
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('TF-IDF + SVM Confusion Matrix')
    
    # Transformer confusion matrix
    sns.heatmap(
        cm_transformer, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names, 
        ax=axes[1]
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Transformer Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Confusion matrices saved to 'confusion_matrices.png'")
    
    # Compare performance by class
    class_comparison = []
    for i, class_name in enumerate(class_names):
        if str(i) in svm_report:
            class_comparison.append({
                'Class': class_name,
                'SVM F1': svm_report[str(i)]['f1-score'],
                'Transformer F1': transformer_report[str(i)]['f1-score'],
                'Difference': transformer_report[str(i)]['f1-score'] - svm_report[str(i)]['f1-score']
            })
    
    class_df = pd.DataFrame(class_comparison)
    print("\nClass-wise F1 score comparison:")
    print(class_df)
    
    # Create bar chart for class-wise performance
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    ax.bar(x - width/2, class_df['SVM F1'], width, label='TF-IDF + SVM')
    ax.bar(x + width/2, class_df['Transformer F1'], width, label='Transformer')
    
    ax.set_xlabel('Sentiment Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Class and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('class_performance.png')
    print("Class performance comparison saved to 'class_performance.png'")
    
    results = {
        'tfidf_svm': {
            'report': svm_report,
            'confusion_matrix': cm_svm,
            'predictions': y_pred_svm
        },
        'transformer': {
            'report': transformer_report,
            'confusion_matrix': cm_transformer,
            'predictions': y_pred_transformer
        },
        'comparison': comparison,
        'class_comparison': class_df
    }
    
    return results


def find_misclassified_examples(X_test, y_test, y_pred_svm, y_pred_transformer, class_names, n=10):
    """
    Find examples of text misclassified by one or both models
    
    Parameters:
    -----------
    X_test : array-like
        Test text data
    y_test : array-like
        True labels
    y_pred_svm : array-like
        SVM predictions
    y_pred_transformer : array-like
        Transformer predictions
    class_names : list
        Class names
    n : int
        Number of examples to return
        
    Returns:
    --------
    dict
        Dictionary with misclassified examples
    """
    misclassified = {
        'both': [],
        'only_svm': [],
        'only_transformer': [],
        'difficult_examples': []
    }
    
    # Find examples misclassified by both models
    both_wrong_indices = np.where((y_test != y_pred_svm) & (y_test != y_pred_transformer))[0]
    for idx in both_wrong_indices[:min(n, len(both_wrong_indices))]:
        misclassified['both'].append({
            'text': X_test[idx],
            'true': class_names[y_test[idx]],
            'svm_pred': class_names[y_pred_svm[idx]],
            'transformer_pred': class_names[y_pred_transformer[idx]]
        })
    
    # Find examples misclassified only by SVM
    svm_wrong_indices = np.where((y_test != y_pred_svm) & (y_test == y_pred_transformer))[0]
    for idx in svm_wrong_indices[:min(n, len(svm_wrong_indices))]:
        misclassified['only_svm'].append({
            'text': X_test[idx],
            'true': class_names[y_test[idx]],
            'svm_pred': class_names[y_pred_svm[idx]]
        })
    
    # Find examples misclassified only by transformer
    transformer_wrong_indices = np.where((y_test == y_pred_svm) & (y_test != y_pred_transformer))[0]
    for idx in transformer_wrong_indices[:min(n, len(transformer_wrong_indices))]:
        misclassified['only_transformer'].append({
            'text': X_test[idx],
            'true': class_names[y_test[idx]],
            'transformer_pred': class_names[y_pred_transformer[idx]]
        })
    
    # Find examples where the two models disagree
    # Find examples where the two models disagree
    disagree_indices = np.where(y_pred_svm != y_pred_transformer)[0]
    for idx in disagree_indices[:min(n, len(disagree_indices))]:
        misclassified['difficult_examples'].append({
            'text': X_test[idx],
            'true': class_names[y_test[idx]],
            'svm_pred': class_names[y_pred_svm[idx]],
            'transformer_pred': class_names[y_pred_transformer[idx]]
        })

    return misclassified

def print_misclassified(misclassified_dict):
    for category, examples in misclassified_dict.items():
        print(f"\n--- {category.upper()} ---")
        for ex in examples:
            print(f"Text: {ex['text']}")
            print(f"True: {ex['true']}")
            if 'svm_pred' in ex:
                print(f"SVM Predicted: {ex['svm_pred']}")
            if 'transformer_pred' in ex:
                print(f"Transformer Predicted: {ex['transformer_pred']}")
            print("-" * 40)
