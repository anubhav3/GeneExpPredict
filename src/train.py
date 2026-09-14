from preprocess import load_data, prepare_data
from model import create_model
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Load data
x, y = load_data()

X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(x, y)



# Create model
model = create_model()


# Loss Function and Optimizer
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Early stopping
best_auc = 0
patience = 10
counter = 0

# Store training history
train_losses = []
val_aucs = []


# Training
for epoch in range(100):

    model.train()

    output = model(X_train).squeeze()

    loss = loss_fn(output, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():

        output = model(X_val).squeeze()

        probability = torch.sigmoid(output)


    val_auc = roc_auc_score(
        Y_val.numpy(),
        probability.numpy()
    )


    # Store results
    train_losses.append(loss.item())
    val_aucs.append(val_auc)


    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {loss.item():.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


    # Early stopping
    if val_auc > best_auc:

        best_auc = val_auc
        counter = 0

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

        print("  → New best model saved!")

    else:

        counter += 1

        print(
            f"  → No improvement "
            f"({counter}/{patience})"
        )


    if counter >= patience:

        print("Early stopping!")
        break

# Plot training history
epochs = range(1, len(train_losses) + 1)


plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label = "Training Loss")
plt.plot(epochs, val_aucs, label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training History")
plt.legend()
plt.savefig("results/figures/training_history.png", dpi=300, bbox_inches="tight")

print(f"Best validation AUC: {best_auc:.4f}")
print("Training plot saved.")