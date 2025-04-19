# src/traditional_model.py
 
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time
from tqdm import tqdm

class TfidfSvmModel:
    """
    Traditional ML model using TF-IDF vectorization and SVM classifier
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF + SVM model
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features for TF-IDF
        ngram_range : tuple
            Range of n-grams to include in TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.classifier = None
        self.training_time = None
        self.inference_time = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=False):
        """
        Train the TF-IDF + SVM model
        
        Parameters:
        -----------
        X_train : array-like
            Training text data
        y_train : array-like
            Training labels
        X_val : array-like, optional
            Validation text data
        y_val : array-like, optional
            Validation labels
        optimize : bool
            Whether to perform hyperparameter optimization
            
        Returns:
        --------
        dict
            Training results
        """
        print("Transforming text data with TF-IDF...")
        start_time = time.time()
        
        # Create TF-IDF vectors
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'class_weight': [None, 'balanced'],
                'dual': [False],
                'max_iter': [1000, 2000]
            }
            
            # Create base SVM model
            base_svm = LinearSVC()
            
            # Grid search
            grid_search = GridSearchCV(
                base_svm, 
                param_grid, 
                cv=3, 
                scoring='f1_macro',
                verbose=1,
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X_train_tfidf, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Create SVM with best parameters
            svm = LinearSVC(**best_params)
        else:
            print("Training SVM with default parameters...")
            # Create default SVM
            svm = LinearSVC(C=1.0, class_weight='balanced')
        
        # Train calibrated classifier
        self.classifier = CalibratedClassifierCV(svm)
        self.classifier.fit(X_train_tfidf, y_train)
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        results = {"training_time": self.training_time}
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("Evaluating on validation set...")
            X_val_tfidf = self.vectorizer.transform(X_val)
            
            inference_start = time.time()
            y_val_pred = self.classifier.predict(X_val_tfidf)
            self.inference_time = (time.time() - inference_start) / len(X_val)
            
            val_report = classification_report(y_val, y_val_pred, output_dict=True)
            results["validation_report"] = val_report
            results["inference_time"] = self.inference_time
            
            print(f"Validation accuracy: {val_report['accuracy']:.4f}")
            print(f"Validation weighted F1: {val_report['weighted avg']['f1-score']:.4f}")
            print(f"Average inference time per sample: {self.inference_time*1000:.2f} ms")
        
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
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Transform texts
        texts_tfidf = self.vectorizer.transform(texts)
        
        # Predict
        return self.classifier.predict(texts_tfidf)
    
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
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Transform texts
        texts_tfidf = self.vectorizer.transform(texts)
        
        # Predict probabilities
        return self.classifier.predict_proba(texts_tfidf)
    
    def save(self, model_path, vectorizer_path):
        """
        Save the model and vectorizer
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        vectorizer_path : str
            Path to save the vectorizer
        """
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load(self, model_path, vectorizer_path):
        """
        Load the model and vectorizer
        
        Parameters:
        -----------
        model_path : str
            Path to load the model from
        vectorizer_path : str
            Path to load the vectorizer from
        """
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Model loaded from {model_path}")
        print(f"Vectorizer loaded from {vectorizer_path}")
 