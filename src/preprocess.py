import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def load_data():

    df_x = pd.read_csv("data/raw/x_train.csv")
    df_y = pd.read_csv("data/raw/y_train.csv")

    return df_x, df_y


def prepare_data(df_x, df_y, save = False):

    # Get unique gene IDs
    ids = df_x["Id"].unique()

    # Split IDs into train and temporary set
    id_train, id_temp = train_test_split(
        ids,
        test_size=0.2,
        random_state=99
    )

    # Split temporary set into validation and test
    id_val, id_test = train_test_split(
        id_temp,
        test_size=0.5,
        random_state=99
    )

    # -------------------------
    # Split X and Y
    # -------------------------

    X_train = df_x[df_x["Id"].isin(id_train)]
    Y_train = df_y[df_y["Id"].isin(id_train)]

    X_val = df_x[df_x["Id"].isin(id_val)]
    Y_val = df_y[df_y["Id"].isin(id_val)]

    X_test = df_x[df_x["Id"].isin(id_test)]
    Y_test = df_y[df_y["Id"].isin(id_test)]
    
    if save:
            X_train.to_csv("data/processed/X_train.csv", index = False)
            Y_train.to_csv("data/processed/Y_train.csv", index = False)
            X_val.to_csv("data/processed/X_val.csv", index = False)
            Y_val.to_csv("data/processed/Y_val.csv", index = False)
            X_test.to_csv("data/processed/X_test.csv", index = False)
            Y_test.to_csv("data/processed/Y_test.csv", index = False)
        

    # -------------------------
    # Remove Id
    # -------------------------

    X_train = X_train.drop(columns="Id").values
    X_val = X_val.drop(columns="Id").values
    X_test = X_test.drop(columns="Id").values

    Y_train = Y_train["Prediction"].values
    Y_val = Y_val["Prediction"].values
    Y_test = Y_test["Prediction"].values

    # -------------------------
    # Reshape
    # -------------------------

    X_train = X_train.reshape(-1, 100, 5)
    X_val = X_val.reshape(-1, 100, 5)
    X_test = X_test.reshape(-1, 100, 5)

    # -------------------------
    # Transpose
    # -------------------------

    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)

    # -------------------------
    # Convert to tensors
    # -------------------------

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    return (
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test
    )
    
    
def main():

    x, y = load_data()
    prepare_data(x, y, save = True)
    print("Preprocessing complete.")
    
if __name__ == "__main__":
        main()