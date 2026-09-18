import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix

from src.model import create_model



# PAGE SETUP
st.set_page_config(page_title="Gene Expression Prediction", page_icon="🧬", layout="wide")

st.title("🧬 Gene Expression Prediction")
st.caption("Predicting gene expression from histone modification signals using CNN")

with st.expander("About this project"):
    st.markdown("""
    This project uses a 1D convolutional neural network (CNN) to predict whether a gene has high or low expression based on five histone modification signals measured across 100 genomic bins around the transcription start site (TSS).
    
    The model learns local patterns across genomic positions while treating the five histone modifications as input channels. The dashboard evaluates the trained model on a held-out test set and provides interactive exploration of individual gene predictions and their histone modification profiles.
    """)
    

# LOAD DATA
X_test = pd.read_csv("data/processed/X_test.csv")
Y_test = pd.read_csv("data/processed/Y_test.csv")



# LOAD MODEL
checkpoint = torch.load("models/best_model.pth", map_location = "cpu")
model_params = checkpoint["model_hyperparameters"]
model = create_model(**model_params)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



# HISTONE COLUMNS
histone_columns = [
    "H3K4me3",
    "H3K4me1",
    "H3K36me3",
    "H3K9me3",
    "H3K27me3"
]



# PREPARE TEST DATA
gene_ids = X_test["Id"].unique()
number_of_genes = len(gene_ids)

X = X_test[histone_columns].values

# (genes * 100, 5) -> (genes, 100, 5)
X = X.reshape(number_of_genes, 100, 5)

# (genes, 100, 5) -> (genes, 5, 100)
X = X.transpose(0, 2, 1)

X = torch.tensor(X, dtype=torch.float32)



# MAKE PREDICTIONS
with torch.no_grad():
    outputs = model(X)
    probabilities = torch.sigmoid(outputs).squeeze().numpy()

predicted_labels = (probabilities >= 0.5).astype(int)



# GET TRUE LABELS


Y_test_sorted = Y_test.set_index("Id").loc[gene_ids]
true_labels = Y_test_sorted["Prediction"].values



# CALCULATE METRICS
test_auc = roc_auc_score(true_labels, probabilities)
test_accuracy = accuracy_score(true_labels, predicted_labels)

number_correct = (predicted_labels == true_labels).sum()
number_incorrect = (predicted_labels != true_labels).sum()



# OVERALL TEST SET PERFORMANCE
st.header("Overall Test Set Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ROC-AUC", f"{test_auc:.3f}")

with col2:
    st.metric("Accuracy", f"{test_accuracy:.1%}")

with col3:
    st.metric("Correct", number_correct)

with col4:
    st.metric("Test Genes", number_of_genes)



# ROC CURVE + PROBABILITY DISTRIBUTION
col1, col2 = st.columns(2)


# ROC CURVE
with col1:

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(true_labels, probabilities)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {test_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test Set ROC Curve")

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# PROBABILITY DISTRIBUTION
with col2:

    st.subheader("Prediction Probability")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.hist(probabilities, bins=15)

    ax.axvline(0.5, linestyle="--", linewidth=1, label="Threshold = 0.5")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Number of genes")
    ax.set_title("Prediction Confidence")

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)



st.subheader("Confusion Matrix")

cm = confusion_matrix(true_labels, predicted_labels)

left, center, right = st.columns([1, 1.2, 1])

with center:

    fig, ax = plt.subplots(figsize=(4, 3.5))

    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Test Set Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["LOW", "HIGH"])
    ax.set_yticklabels(["LOW", "HIGH"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)



# ALL TEST GENE PREDICTIONS
st.header("All Test Gene Predictions")

prediction_table = pd.DataFrame({
    "Gene ID": gene_ids,
    "Probability": probabilities,
    "Predicted": np.where(predicted_labels == 1, "HIGH", "LOW"),
    "Observed": np.where(true_labels == 1, "HIGH", "LOW"),
    "Correct": np.where(predicted_labels == true_labels, "✓", "✗")
})

st.dataframe(prediction_table, use_container_width=True, hide_index=True)



# DOWNLOAD PREDICTIONS
csv = prediction_table.to_csv(index=False)

st.download_button(
    label="Download predictions",
    data=csv,
    file_name="test_predictions.csv",
    mime="text/csv"
)



# INDIVIDUAL GENE ANALYSIS
st.header("Individual Gene Analysis")

selected_gene = st.selectbox("Select Gene ID", sorted(gene_ids))

gene_data = X_test[X_test["Id"] == selected_gene].copy()

gene_index = np.where(gene_ids == selected_gene)[0][0]

gene_probability = probabilities[gene_index]

gene_prediction = "HIGH" if gene_probability >= 0.5 else "LOW"

gene_observed_value = true_labels[gene_index]

gene_observed = "HIGH" if gene_observed_value == 1 else "LOW"



# INDIVIDUAL GENE SUMMARY
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Probability", f"{gene_probability:.1%}")

with col2:
    st.metric("Predicted Expression", gene_prediction)

with col3:
    st.metric("Observed Expression", gene_observed)


if gene_prediction == gene_observed:
    st.success("✓ Prediction is correct")
else:
    st.error("✗ Prediction is incorrect")



# HISTONE MODIFICATION SIGNALS
st.subheader("Histone Modification Signals")


# ------------------------------------------------------------
# H3K4me3 + H3K4me1
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.plot(gene_data["H3K4me3"].values)

    ax.set_title("H3K4me3")
    ax.set_xlabel("Genomic Bin")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


with col2:

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.plot(gene_data["H3K4me1"].values)

    ax.set_title("H3K4me1")
    ax.set_xlabel("Genomic Bin")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# ------------------------------------------------------------
# H3K36me3 + H3K9me3
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.plot(gene_data["H3K36me3"].values)

    ax.set_title("H3K36me3")
    ax.set_xlabel("Genomic Bin")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


with col2:

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.plot(gene_data["H3K9me3"].values)

    ax.set_title("H3K9me3")
    ax.set_xlabel("Genomic Bin")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# ------------------------------------------------------------
# H3K27me3
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.plot(gene_data["H3K27me3"].values)

    ax.set_title("H3K27me3")
    ax.set_xlabel("Genomic Bin")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)