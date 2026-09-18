# Train and hypertune the CNN model with MLFlow
from preprocess import load_data, prepare_data
from model import create_model
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from itertools import product



# SETTINGS

learning_rates = [0.0001, 0.0003, 0.001]
filters = [16, 32, 64] #Controls the nr. of features the CNN learns from the genomic sequence
kernel_sizes = [3, 5, 7] #A convolutional kernel looks at neighbouring bins. How much local genomic context is useful.
dense_sizes = [16, 32, 64] #Controls the size of the fully connected layer after the CNN.

EPOCHS = 100 #Max number of passes through the training data.
PATIENCE = 10 #For early stopping



# LOAD DATA
x, y = load_data()
X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(x, y)



# MLflow
mlflow.set_experiment("gene-expression-cnn")



# HYPERPARAMETER SEARCH
combinations = product(learning_rates, filters, kernel_sizes, dense_sizes)

for lr, num_filters, kernel_size, dense_size in combinations:

    print("\n----------------------------------------")
    print(f"LR: {lr} | Filters: {num_filters} | Kernel: {kernel_size} | Dense: {dense_size}")
    print("----------------------------------------")

    with mlflow.start_run():

        mlflow.log_params({"learning_rate": lr, "filters": num_filters, "kernel_size": kernel_size, "dense_size": dense_size, "epochs": EPOCHS, "patience": PATIENCE})

        model = create_model(filters=num_filters, kernel_size=kernel_size, dense_size=dense_size)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_auc = 0
        counter = 0

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

            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                print("Early stopping.")
                break

        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.log_metric("epochs_trained", epoch + 1)

        mlflow.pytorch.log_model(model, name="model", input_example=X_train[:1], serialization_format="pickle")

        print(f"Best Val AUC: {best_auc:.4f}")

