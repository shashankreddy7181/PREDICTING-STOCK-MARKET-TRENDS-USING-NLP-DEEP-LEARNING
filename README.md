# 📈 NLP-Driven Stock Market Prediction Using Deep Learning

## 👥 Team Members

* Rishik Vardhan Reddy U
* Shashank Reddy D
* Siva Teja Katta
* Maneesh Konda

A hybrid AI-based stock market prediction system that combines **technical analysis**, **Natural Language Processing (NLP)**, and **deep learning models** to predict stock price movement more accurately.

This project integrates:

* Historical stock market data
* Financial news sentiment analysis
* Technical indicators
* Deep learning models (LSTM)
* Ensemble machine learning techniques

The goal is to bridge the gap between traditional numerical forecasting and real-world market sentiment.

---

# 🚀 Project Overview

Traditional stock prediction systems rely mainly on historical price data. However, real stock markets are heavily influenced by:

* Financial news
* Political events
* Company announcements
* Market sentiment
* Economic conditions

This project improves prediction performance by combining:

✅ Technical indicators from stock prices
✅ NLP-based sentiment analysis from financial news
✅ Deep learning sequence models
✅ Ensemble learning methods

The final system compares multiple models and evaluates their performance using classification metrics.

---

# 🧠 Models Implemented

## 1.Baseline LSTM

Uses only historical stock price data.

### Architecture:

* LSTM layers
* Batch Normalization
* Dropout
* Dense layers

---

## 2.Hybrid Fusion Model

Combines:

* Stock price sequences
* NLP sentiment features

### NLP Features:

* News Sentiment Score
* News Volume
* Named Entity Recognition (NER) Score
* TF-IDF Features

This model learns both:

* Numerical market trends
* External sentiment influence

---

## 3.Tree Ensemble

Uses ensemble machine learning with NLP-enhanced features.

### Algorithm:

* HistGradientBoostingClassifier

---

## 4.Multi-Model Soft Voting Ensemble

Combines predictions from:

* Baseline LSTM
* Hybrid Fusion Model
* NLP Tree Ensemble

Uses probability averaging for final predictions.

---


# ⚙️ Technologies Used

## Programming Language

* Python

## Libraries & Frameworks

### Data Processing

* pandas
* numpy

### NLP

* nltk
* VADER Sentiment Analyzer
* TF-IDF Vectorizer

### Deep Learning

* TensorFlow

### Machine Learning

* scikit-learn

### Data Collection

* yfinance
* datasets (Hugging Face)

### Visualization

* matplotlib
* seaborn

---

# 🔄 Project Workflow

## 1. Fetch Stock Prices

Historical stock data is downloaded using Yahoo Finance.

## 2. Fetch Financial News

Financial news articles are collected from Hugging Face datasets.

## 3. Feature Engineering

Technical indicators are generated, including:

* RSI
* Moving Averages
* Price Momentum
* Volatility indicators

## 4. NLP Feature Engineering

News headlines are processed using:

* VADER sentiment analysis
* TF-IDF vectorization
* Named Entity Recognition scoring

## 5. Data Alignment

Stock data and news data are aligned by date while avoiding look-ahead bias.

## 6. Sequence Generation

Time-series sequences are created for LSTM training.

## 7. Model Training

All models are trained independently.

## 8. Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

## 9. Visualization

Performance comparison charts and confusion matrices are generated.

---

# 📈 Evaluation Metrics

The following metrics are used for model comparison:

| Metric    | Description                          |
| --------- | ------------------------------------ |
| Accuracy  | Overall prediction correctness       |
| Precision | Positive prediction quality          |
| Recall    | Ability to detect positive cases     |
| F1-Score  | Balance between precision and recall |
| ROC-AUC   | Classification confidence quality    |

---

# 📂 Project Structure

```bash
├── stocks.ipynb                # Main project notebook
├── saved_models/               # Saved trained models
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then run:

```bash
stocks.ipynb
```

---

# 📊 Key Findings

* Hybrid models outperform traditional price-only models.
* Sentiment analysis improves prediction capability.
* Financial news significantly impacts short-term market movement.
* Ensemble learning provides more stable performance.
* Combining NLP with deep learning improves overall robustness.

---

# 🔮 Future Improvements

Potential future enhancements:

* Real-time news API integration
* Transformer-based NLP models (BERT/GPT)
* Social media sentiment analysis (Twitter/Reddit)
* Reinforcement learning for trading strategies
* Live dashboard deployment using Streamlit
* Multi-market global prediction support

---

# 📸 Visualizations Included

The notebook contains:

* Model comparison charts
* Confusion matrices
* Correlation heatmaps
* Prediction trend plots
* Performance evaluation graphs

---
