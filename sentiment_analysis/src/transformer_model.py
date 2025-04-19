# src/transformer_model.py
 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
from tqdm import tqdm
import json


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis with transformers"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize dataset
        
        Parameters:
        -----------
        texts : array-like
            Input texts
        labels : array-like
            Input labels
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer for encoding texts
        max_length : int
            Maximum sequence length
        """
        self.encodings = tokenizer(
            list(texts), 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        """Get item by index"""
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Get dataset length"""
        return len(self.labels)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.
    
    Parameters:
    -----------
    eval_pred : tuple
        Contains logits and labels
        
    Returns:
    --------
    dict
        Dictionary containing metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted',  # Use 'binary' for binary classification
        zero_division=0
    )
    
    # Return as dictionary (metric names must match metric_for_best_model if specified)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_metricsS(pred):
    """
    Compute evaluation metrics for transformer model
    
    Parameters:
    -----------
    pred : transformers.EvalPrediction
        Prediction object containing predictions and labels
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metricsss(pred):
    # 1. Extract ground-truth labels
    labels = pred.label_ids

    # 2. Extract logits (handle tuple or list)
    raw_preds = pred.predictions
    logits = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds

    # 3. If logits is a list of arrays, stack them
    if isinstance(logits, list):
        # assumes each element has shape (num_labels,)
        logits = np.stack(logits, axis=0)

    # 4. Compute predicted class indices
    preds = np.argmax(logits, axis=-1)

    # 5. Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metricsd(pred):
    # Extract labels
    labels = pred.label_ids
    
    # Handle tuple outputs (e.g., logits, hidden states)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions  # :contentReference[oaicite:7]{index=7}
    
    # Convert to NumPy array
    logits = np.array(logits)  # :contentReference[oaicite:8]{index=8}
    
    # Obtain predicted class indices
    preds = np.argmax(logits, axis=-1)  # :contentReference[oaicite:9]{index=9}
    
    # Compute core metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class TransformerModel:
    """Transformer-based model for sentiment analysis"""
    
    def __init__(self, model_name="bert-base-uncased", num_labels=3):
        """
        Initialize transformer model
        
        Parameters:
        -----------
        model_name : str
            Pretrained model name
        num_labels : int
            Number of sentiment classes
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.trainer = None
        self.training_time = None
        self.inference_time = None
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def train(self, X_train, y_train, X_val, y_val, output_dir="./transformer_results", 
              batch_size=16, epochs=3, learning_rate=5e-5):
        """
        Train the transformer model
        
        Parameters:
        -----------
        X_train : array-like
            Training text data
        y_train : array-like
            Training labels
        X_val : array-like
            Validation text data
        y_val : array-like
            Validation labels
        output_dir : str
            Directory to save model outputs
        batch_size : int
            Training batch size
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
            
        Returns:
        --------
        dict
            Training results
        """
        print("Preparing datasets for transformer model...")
        
        # Create datasets
        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer)
        val_dataset = SentimentDataset(X_val, y_val, self.tokenizer)
        
         
        # training_argss = TrainingArguments(
        #     output_dir=output_dir,
        #     num_train_epochs=epochs,
        #     fp16=False,
        #     per_device_train_batch_size=batch_size,
        #     dataloader_num_workers=4,
        #     learning_rate=learning_rate,
        #     lr_scheduler_type="cosine",
        #     weight_decay=1e-4,
        #     max_grad_norm=0.01,
        #     metric_for_best_model="eval_map",
        #     greater_is_better=True,
        #     load_best_model_at_end=True,
        #     eval_strategy="epoch",
        #     save_strategy="epoch",
        #     save_total_limit=2,
        #     remove_unused_columns=False,
        #     eval_do_concat_batches=False,
        #     push_to_hub=False,
        #     )
        
        training_args = TrainingArguments( output_dir=output_dir ,
                                          evaluation_strategy="epoch",
                                learning_rate=2e-5,
                                # per_device_train_batch_size=8,
                                # per_device_eval_batch_size=8,
                                num_train_epochs=3,
                                weight_decay=0.01 
                                           )

        
        # Create trainer
        self.trainer = Trainer(
            model = self.model,
            args=training_args,
              train_dataset=train_dataset,
            eval_dataset=val_dataset 
                )
        
 
        # Train model
        print("Training transformer model...")
        start_time = time.time()
        train_result = self.trainer.train()
        self.training_time = time.time() - start_time
        
        # Save training result to a file
        result_path = "training_result.json"
        with open(result_path, "w") as result_file:
            json.dump(train_result, result_file)

        # Evaluate
        print("Evaluating transformer model...")
        eval_result = self.trainer.evaluate()
        
        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():
            self.model.eval()
            predictions = self.trainer.predict(val_dataset)
        self.inference_time = (time.time() - inference_start) / len(X_val)
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        # print(f"Validation accuracy: {eval_result['eval_accuracy']:.4f}")
        # print(f"Validation weighted F1: {eval_result['eval_f1']:.4f}")
        print(f"Average inference time per sample: {self.inference_time*1000:.2f} ms")
        
        results = {
            "training_time": self.training_time,
            # "validation_results": eval_result,
            "inference_time": self.inference_time
        }
        
        return results
    
    def predict(self, texts):
        """
        Predict sentiment for new texts
        
        Parameters:
        -----------
        texts : array-like
            Input texts for prediction
            
        Returns:
        --------
        array
            Predicted labels
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create dataset
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer)
        
        # Make predictions
        predictions = self.trainer.predict(dataset)
        
        return predictions.predictions.argmax(-1)
    
    def predict_proba(self, texts):
        """
        Predict sentiment probabilities for new texts
        
        Parameters:
        -----------
        texts : array-like
            Input texts for prediction
            
        Returns:
        --------
        array
            Predicted class probabilities
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create dataset
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer)
        
        # Make predictions
        predictions = self.trainer.predict(dataset)
        
        # Convert logits to probabilities using softmax
        return torch.nn.functional.softmax(
            torch.from_numpy(predictions.predictions), dim=-1
        ).numpy()
    
    def save(self, output_dir):
        """
        Save the model and tokenizer
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model and tokenizer
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
    
    def load(self, model_dir):
        """
        Load the model and tokenizer
        
        Parameters:
        -----------
        model_dir : str
            Directory from which to load model and tokenizer
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"Model and tokenizer loaded from {model_dir}")
 
