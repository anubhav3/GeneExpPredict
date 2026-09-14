import torch.nn as nn

def create_model():
    
    model = nn.Sequential(
        # Input: 5 x 100
        nn.Conv1d(
            in_channels = 5,
            out_channels = 32,
            kernel_size = 5,
            padding = 2
        ),
        nn.ReLU(),
        # 100 -> 50
        nn.MaxPool1d(kernel_size = 2),
        # 32 x 50 -> 1600
        nn.Flatten(),
        # 1600 -> 32
        nn.Linear(32*50, 32),
        nn.ReLU(),
        nn.Linear(32, 1)    
    )
    
    return model
