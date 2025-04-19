#data sampling.py


import pandas as pd

def create_sample_csv(file_path):
    df = pd.read_csv(file_path)
    positive_df = df[df['sentiment'] == 'positive'].head(100)
    negative_df = df[df['sentiment'] == 'negative'].head(100)
    sample_df = pd.concat([positive_df, negative_df])
    sample_df.to_csv('sample100.csv', index=False)

create_sample_csv(file_path="/Users/pruthvirajadhav/code/AI assignment/mycoursera/sentiment_analysis/data/IMDB Dataset.csv")
