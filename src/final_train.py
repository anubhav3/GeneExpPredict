# Train the CNN model using the final hyperparameters

from preprocess import load_data, prepare_data
from model import create_model
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import copy


# Final hyperparameters
LEARNING_RATE = 0.001
FILTERS = 16
KERNEL_SIZE = 3
DENSE_SIZE = 64
EPOCHS = 100
PATIENCE = 10


# Load data
x, y = load_data()
X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(x, y)


# Create model
model = create_model(filters = FILTERS, kernel_size = KERNEL_SIZE, dense_size = DENSE_SIZE)


# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


# Training
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


# Load best model
model.load_state_dict(best_state)
model.eval()


# Save model and metadata
checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_hyperparameters": {
        "filters": FILTERS,
        "kernel_size": KERNEL_SIZE,
        "dense_size": DENSE_SIZE
    },
    "training_hyperparameters": {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "patience": PATIENCE
    },
    "best_val_auc": best_auc
}

torch.save(checkpoint, "models/best_model.pth")


# Summary
print(f"\nBest validation AUC = {best_auc:.4f}")
print("Best model and metadata saved to models/best_model.pth")

checkpoint = torch.load("models/best_model.pth", map_location = "cpu")

model_params = checkpoint["model_hyperparameters"]

model = create_model(**model_params)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

