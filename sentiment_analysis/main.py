"""
# main.py
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Import modules
from src.data_processing import load_data, prepare_data
from src.traditional_model import TfidfSvmModel
from src.transformer_model import TransformerModel
from src.evaluation import evaluate_and_compare, find_misclassified_examples
 


def main(args):
    """Main execution function"""
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Load data
    args.data_path = "/Users/pruthvirajadhav/code/AI assignment/mycoursera/sentiment_analysis/data/sample100.csv"
    df = load_data(args.data_path)
    
    # Check if data has the required columns
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in the dataset")
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in the dataset")
    
    # Prepare data
    data = prepare_data(
        df, 
        args.text_column, 
        args.label_column,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Get class names
    class_names = list(data['label_map'].values())
    print(f"Classes: {class_names}")
    
    # Train traditional ML model
    print("\n" + "="*50)
    print("Training TF-IDF + SVM model...")
    print("="*50)
    tfidf_svm_model = TfidfSvmModel(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max)
    )
    tfidf_svm_results = tfidf_svm_model.train(
        data['X_train'], 
        data['y_train'],
        data['X_val'],
        data['y_val'],
        optimize=args.optimize
    )
    
    # Save traditional model
    if args.save_models:
        tfidf_svm_model.save(
            os.path.join(args.output_dir, 'models', 'svm_model.joblib'),
            os.path.join(args.output_dir, 'models', 'tfidf_vectorizer.joblib')
        )
    
    # Train transformer model if requested
    transformer_model = None
    if not args.skip_transformer:
        print("\n" + "="*50)
        print("Training Transformer model...")
        print("="*50)
        transformer_model = TransformerModel(
            model_name=args.transformer_model,
            num_labels=len(class_names)
        )
        transformer_results = transformer_model.train(
            data['X_train'], 
            data['y_train'],
            data['X_val'],
            data['y_val'],
            output_dir=os.path.join(args.output_dir, 'models', 'transformer'),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Save transformer model
        if args.save_models:
            transformer_model.save(os.path.join(args.output_dir, 'models', 'transformer_final'))
    
    # Evaluate models
    print("\n" + "="*50)
    print("Evaluating models...")
    print("="*50)
    
    if transformer_model:
        # Compare both models
        eval_results = evaluate_and_compare(
            data['X_test'], 
            data['y_test'],
            tfidf_svm_model,
            transformer_model,
            class_names
        )
        
        # Find misclassified examples
        misclassified = find_misclassified_examples(
            data['X_test'],
            data['y_test'],
            eval_results['tfidf_svm']['predictions'],
            eval_results['transformer']['predictions'],
            class_names
        )
        
        # Save misclassified examples
        misclassified_df = pd.DataFrame(misclassified['both'])
        misclassified_df.to_csv(
            os.path.join(args.output_dir, 'results', 'both_misclassified.csv'),
            index=False
        )
    else:
        # Evaluate only SVM model
        y_pred_svm = tfidf_svm_model.predict(data['X_test'])
        svm_report = classification_report(data['y_test'], y_pred_svm, target_names=class_names)
        print("TF-IDF + SVM Test Results:")
        print(svm_report)
    
    # Generate explanations if requested
    if args.explain:
        print("\n" + "="*50)
        print("Generating model explanations...")
        print("="*50)
        
    
        
 
        
        
    
    print("\n" + "="*50)
    print("Sentiment analysis completed successfully!")
    print("="*50)

# if __name__ == "__main__":
    # print("hi")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentiment Analysis System")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset CSV file')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='sentiment',
                        help='Column name for sentiment labels')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.5,
                        help='Proportion of non-training data to use for validation')
    
    # TF-IDF + SVM parameters
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum number of features for TF-IDF')
    parser.add_argument('--ngram_max', type=int, default=2,
                        help='Maximum n-gram size')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization for SVM')
    
    # Transformer parameters
    parser.add_argument('--skip_transformer', action='store_true',
                        help='Skip training the transformer model')
    parser.add_argument('--transformer_model', type=str, default='bert-base-uncased',
                        help='Pretrained transformer model name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for transformer training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for transformer training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for transformer training')
    
    # Explanation parameters
    parser.add_argument('--explain', action='store_true',
                        help='Generate model explanations')
    parser.add_argument('--explain_text', type=str,
                        help='Text to explain (if not provided, a sample will be used)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--save_models', action='store_true',
                        help='Save trained models')
    
    args = parser.parse_args()
    main(args)
 

# data/sample_data.csv (example format)
"""
text,sentiment
This movie was amazing! I loved every minute of it.,positive
The product arrived damaged and customer service was unhelpful.,negative
It's an okay service but nothing special.,neutral
I'm very disappointed with the quality.,negative
The staff was friendly and helpful.,positive
The room was clean but small.,neutral
"""