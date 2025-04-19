# README.md
"""
# Sentiment Analysis System

This project implements a sentiment analysis system that classifies text into positive, negative, or neutral categories using both traditional machine learning (TF-IDF + SVM) and transformer-based approaches.

## Features
- Text preprocessing and cleaning
- Traditional ML baseline using TF-IDF + SVM
- Transformer-based classifier (BERT)
- Model evaluation and comparison
- Prediction explanations

## Setup
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Place your dataset in the `data/` directory
3. Run the main script:
   ```
   python main.py
   ```

## Project Structure
- `src/data_processing.py`: Text preprocessing and data handling
- `src/traditional_model.py`: TF-IDF + SVM implementation
- `src/transformer_model.py`: BERT transformer implementation
- `src/evaluation.py`: Model evaluation metrics and comparison
- `src/explainability.py`: Methods to explain model predictions
- `main.py`: Main execution script

## Requirements
- Python 3.8+
- See requirements.txt for dependencies
"""