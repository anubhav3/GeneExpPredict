import torch
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from preprocess import load_data, prepare_data
from model import create_model


# LOAD DATA
x, y = load_data()
X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(x, y)


# LOAD BEST MODEL
model = create_model(filters = 16, kernel_size = 3, dense_size = 64)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()


# MAKE PREDICTIONS
with torch.no_grad():
    test_output = model(X_test).squeeze()
    test_probability = torch.sigmoid(test_output)


# TEST ROC-AUC
auc = roc_auc_score(Y_test.numpy(), test_probability.numpy())
print(f"Test ROC-AUC = {auc:.4f}")


# TEST ACCURACY
test_prediction = (test_probability >= 0.5).float()
accuracy = accuracy_score(Y_test.numpy(), test_prediction.numpy())
print(f"Test Accuracy = {accuracy:.4f}")


# CONFUSION MATRIX
matrix = confusion_matrix(Y_test.numpy(), test_prediction.numpy())

print("\nConfusion Matrix:")
print(matrix)