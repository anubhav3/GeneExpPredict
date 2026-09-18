import torch.nn as nn

def create_model(filters=32, kernel_size=5, dense_size=32):

    model = nn.Sequential(
        nn.Conv1d(in_channels=5, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Flatten(),
        nn.LazyLinear(dense_size),
        nn.ReLU(),
        nn.Linear(dense_size, 1)
    )

    return model
