# GeneExpPredict 🧬

A deep learning project for predicting gene expression outcomes from genomic sequence data using a **1D Convolutional Neural Network (CNN)**.

### 🚀 Live Demo

**[Try GeneExpPredict →](https://geneexppredict.streamlit.app/)**

## 🔍 Overview

GeneExpPredict takes encoded DNA sequence data and uses a CNN to learn sequence patterns associated with gene expression.

The project includes:

* 🧬 DNA sequence preprocessing and encoding
* 🧠 CNN-based binary classification
* 🔧 Hyperparameter tuning with MLflow
* 📊 Validation using ROC-AUC
* 💾 Saved model checkpoint with architecture and training parameters
* 🚀 Interactive Streamlit application

## 🛠️ Tech Stack

**Python · PyTorch · scikit-learn · MLflow · Pandas · Streamlit**

## 📁 Project Structure

```text
GeneExpPredict/
├── app/
│   └── app.py
├── data/
│   └── processed/
├── models/
│   └── best_model.pth
├── src/
│   ├── model.py
│   └── preprocess.py
├── final_train.py
└── requirements.txt
```

## 🚀 Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run app/app.py
```

## 🎯 Goal

GeneExpPredict demonstrates an end-to-end machine learning workflow — from **genomic data preprocessing and model development to evaluation and deployment**.

---

*Built with Python and PyTorch.*
