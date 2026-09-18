# Train the CNN model, include the final hyper parameters extracted from train.py
from preprocess import load_data, prepare_data
from model import create_model
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import copy



# These hyperparameters are finalised from hypertuning using MlFlow train.py script
LEARNING_RATE = 0.001
FILTERS = 16
KERNEL_SIZE = 3
DENSE_SIZE = 64
EPOCHS = 100
PATIENCE = 10


# LOAD DATA
x, y = load_data()
X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(x, y)



# CREATE MODEL
model = create_model(filters = FILTERS, kernel_size = KERNEL_SIZE, dense_size = DENSE_SIZE)



# LOSS AND OPTIMIZER
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)



# TRAINING
best_auc = 0
counter = 0
best_state = None

for epoch in range(EPOCHS):

    model.train()

    output = model(X_train).squeeze()
    loss = loss_fn(output, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        output = model(X_val).squeeze()
        probability = torch.sigmoid(output)

    val_auc = roc_auc_score(Y_val.numpy(), probability.numpy())

    print(f"Epoch {epoch + 1} | Loss = {loss.item():.4f} | Val AUC = {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        counter = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        counter += 1

    if counter >= PATIENCE:
        print("Early stopping.")
        break



# LOAD BEST MODEL
model.load_state_dict(best_state)
model.eval()


# SAVE BEST MODEL
torch.save(model.state_dict(), "models/best_model.pth")

print(f"\nBest validation AUC = {best_auc:.4f}")
print("Best model saved to models/best_model.pth")
