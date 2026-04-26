# Fake News Detection System

##  Overview
A machine learning system that classifies news articles as FAKE or REAL using Natural Language Processing techniques.

##  Dataset
- **Source**: [Kaggle Fake News Dataset] (https://www.kaggle.com/datasets/hassanamin/textdb3)
- **Size**: 6,335 articles (3,164 FAKE, 3,171 REAL)
- **Features**: News title, article text, and label

## 🛠️ Technologies
- Python 3.13
- scikit-learn (TF-IDF, Logistic Regression)
- TensorFlow/Keras (LSTM)
- NLTK (text preprocessing)
- Gensim (GloVe embeddings)

## 📈 Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression (TF-IDF) | 91.40% |
| Bi-LSTM (GloVe) | 91.55% |
