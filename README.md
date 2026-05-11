# Fake News Detection System

## Overview
A machine learning system that classifies news articles as **FAKE** or **REAL** using Natural Language Processing techniques, featuring three different models ranging from traditional machine learning to state-of-the-art transformers.

## Dataset
- **Source**: [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/hassanamin/textdb3)
- **Size**: 6,335 articles (3,164 FAKE, 3,171 REAL)
- **Features**: News title, article text, and label

## 🛠️ Technologies
| Category | Technologies |
|----------|--------------|
| **Core ML** | scikit-learn (TF-IDF, Logistic Regression) |
| **Deep Learning** | TensorFlow/Keras (Bi-LSTM) |
| **Transformers** | Hugging Face Transformers (DistilBERT) |
| **Embeddings** | GloVe (100d), BERT embeddings |
| **Text Processing** | NLTK (tokenization, stopwords, stemming) |
| **Visualization** | Matplotlib, Seaborn, WordCloud |

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (TF-IDF) | 91.40% | 92.06% | 90.77% | 91.41% |
| Bi-LSTM (GloVe) | 91.95% | 92.02% | 92.02% | 92.02% |
| **BERT (DistilBERT)** | **96.61%** | **95.71%** | **97.65%** | **96.67%** |

## Key Insights
- **FAKE news** commonly uses words like: *clinton, trump, hillari, elect*
- **REAL news** commonly uses words like: *said, state, republican, presid*
- Transformer-based models (BERT) significantly outperform traditional approaches

## Project Structure
FakeNewsDetection/
├── FakeNewsDetectionProject.ipynb # Main Jupyter notebook
├── requirements.txt # Python dependencies
├── model.pkl # Saved Logistic Regression model
├── vectorizer.pkl # TF-IDF vectorizer
├── lstm_model.h5 # Saved LSTM model
├── tokenizer.pkl # LSTM tokenizer
├── bert_model/ # Saved BERT model directory
├── bert_tokenizer/ # Saved BERT tokenizer
├── label_encoder.pkl # Label encoder for BERT
└── README.md # This file


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/sondosmhmd/FakeNewsDetection.git
cd FakeNewsDetection

### 2. Install dependencies
pip install -r requirements.txt

###3. Download NLTK data
import nltk
nltk.download('stopwords')
nltk.download('punkt')


###  4. Download GloVe embeddings
bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip


##Usage
Run the complete pipeline
Open and run FakeNewsDetectionProject.ipynb in Jupyter Notebook.
