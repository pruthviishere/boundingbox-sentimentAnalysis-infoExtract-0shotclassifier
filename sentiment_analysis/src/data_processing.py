# src/data_processing.py
 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# # Download NLTK resources
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('wordnet')


def load_data(file_path):
    """
    Load sentiment data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def preprocess_text(text):
    """
    Clean and preprocess text data
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    # tokens = word_tokenize(text)
    # Remove stopwords (optional, as they might carry sentiment)
    # tokens = [word for word in tokens if word not in set(stopwords.words('english'))]
    # Lemmatize
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to text
    # processed_text = ' '.join(tokens)
    return text


def prepare_data(df, text_column, label_column, test_size=0.2, val_size=0.5):
    """
    Prepare data for training and evaluation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str
        Column containing text data
    label_column : str
        Column containing sentiment labels
    test_size : float
        Proportion of data to use for testing
    val_size : float
        Proportion of non-training data to use for validation
        
    Returns:
    --------
    dict
        Dictionary containing processed data splits
    """
    print("Preprocessing text data...")
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Map string labels to numeric if needed
    if df[label_column].dtype == 'object':
        print("Converting labels to numeric...")
        # Assuming labels are 'negative', 'neutral', 'positive'
        # label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label_map = {'negative': 0, 'positive': 1}
        
        # Try to map known labels, if not in map, default to mapping values
        try:
            df['sentiment_id'] = df[label_column].map(label_map)
        except:
            # Get unique values and create mapping
            unique_labels = df[label_column].unique()
            custom_label_map = {label: idx for idx, label in enumerate(unique_labels)}
            df['sentiment_id'] = df[label_column].map(custom_label_map)
            print(f"Created custom label mapping: {custom_label_map}")
    else:
        df['sentiment_id'] = df[label_column]
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    
    # First split: training and temporary (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['processed_text'].values, 
        df['sentiment_id'].values,
        test_size=test_size, 
        random_state=42, 
        stratify=df['sentiment_id'].values
    )
    
    # Second split: validation and test from temporary data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, 
        y_temp,
        test_size=val_size, 
        random_state=42, 
        stratify=y_temp
    )
    
    print(f"Data split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")
    
    # Create a dictionary with all data
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_map': {0: 'negative', 1: 'neutral', 2: 'positive'} if 'label_map' not in locals() else {v: k for k, v in label_map.items()}
    }
    
    return data
 
