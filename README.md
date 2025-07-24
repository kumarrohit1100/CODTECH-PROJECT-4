# ✈️ NLP: Sentiment Analysis of Airline Tweets

A complete natural language processing (NLP) pipeline to classify customer sentiment in real airline tweets, using Python and standard machine learning techniques.

---

## 📁 **CODTECH-PROJECT-4**

---

## 🛠 Tech Stack

- Python 3
- pandas, matplotlib
- scikit-learn
- NLTK (Natural Language Toolkit)

---

## 📌 Project Description

This project analyzes customer sentiments from real Twitter data related to airline experiences. The goal is to identify whether a tweet expresses **positive**, **neutral**, or **negative** sentiment. The pipeline includes comprehensive **text preprocessing**, **model building**, **evaluation**, and **comparative analysis** to determine the most effective sentiment classifier.

---

## 🚀 Features

### 🧹 Text Preprocessing
- Tokenization
- Stopword removal
- Stemming using NLTK
- CountVectorizer and TF-IDF transformation

### 🤖 ML Models Used
- Multinomial Naive Bayes
- Logistic Regression
- LinearSVC (Support Vector Classifier)

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Visuals

### 🔁 Comparative Analysis
- Evaluated each model with both **CountVectorizer** and **TF-IDF**
- Summarized results in performance tables
- Used sample confusion matrices for interpretability

---

## 📈 Example Results Table

| Model               | Vectorizer | Accuracy | F1 Score |
|---------------------|------------|----------|----------|
| Logistic Regression | TF-IDF     | 0.786    | 0.774    |
| Linear SVC          | TF-IDF     | 0.772    | 0.766    |

✅ **Best Result**: `Logistic Regression` using `TF-IDF` vectorization gave the **highest accuracy of 78.6%** and a solid F1-score of **0.774**.

---

## 🔍 Project Highlights

- 📦 End-to-end NLP pipeline
- 📈 Visualizations: Confusion matrix plots and model comparison charts
- 📚 Clean, modular, and reusable codebase
- 🚀 Ready-to-use baseline template for text classification tasks

---

## 📓 Internship Summary & Learning Outcomes

### 💡 Technical Exposure
- Hands-on experience with **NLP preprocessing**, **model selection**, and **evaluation metrics**
- Exposure to **data pipelines**, **feature extraction**, and **model tuning**

### 🧠 Business Value
- Provides a quick and scalable solution for sentiment monitoring
- Helps companies improve **customer support** and **brand awareness**

### 🤝 Collaboration
- Version-controlled using **GitHub**
- Built for **reproducibility and team collaboration**

---

## 🔧 Next Steps

- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Advanced NLP techniques: Word2Vec, GloVe, transformers
- Deployment as a REST API or Streamlit dashboard
- Automation of preprocessing and model updates

---

## 📂 Project Structure

