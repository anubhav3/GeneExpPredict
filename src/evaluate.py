from turtle import mode
import torch
from sklearn.metrics import roc_auc_score

from preprocess import load_data, prepare_data
from model import create_model

# Load data
x, y = load_data()
X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(x, y)

# Load model
model = create_model()
model.load_state_dict(torch.load("models/cnn_model.pth"))
model.eval()

# Make predictions
with torch.no_grad():
    
    # Raw model output
    test_output = model(X_test).squeeze()
    
    # Convert logits -> Probabilities
    test_probability = torch.sigmoid(test_output)
    
    
# Calculate AUC

auc = roc_auc_score(Y_test, test_probability.numpy())
print(f"Test ROC-AUC: {auc:.4f}")
