
Model Comparison:
                     TF-IDF + SVM  Transformer
Precision                0.853535     0.250000
Recall                   0.850000     0.500000
F1-Score                 0.849624     0.333333
Accuracy                 0.850000     0.500000
Training Time (s)        0.106954     8.173328
Inference Time (ms)      0.092196     3.362048
Confusion matrices saved to 'confusion_matrices.png'

Class-wise F1 score comparison:
      Class    SVM F1  Transformer F1  Difference
0  negative  0.857143        0.000000   -0.857143
1  positive  0.842105        0.666667   -0.175439
Class performance comparison saved to 'class_performance.png'

==================================================
Generating model explanations...
==================================================

==================================================
Sentiment analysis completed successfully!