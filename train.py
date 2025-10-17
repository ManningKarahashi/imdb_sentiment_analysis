import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from dataset import IMDBDataset, collate_fn
from model import LSTMSentiment
import os

MODEL_PATH = "models/imdb_lstm_model.pth"

def train_model(model, train_loader, device, epochs = 5, lr = 0.001, save_path = None):
    cost_function = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        model.train()                                                                           # Set model to training mode
        loop = tqdm(train_loader, desc = f"Epoch {epoch + 1}")                                  # Progress bar for each epoch
        total_cost = 0
        for texts, labels in loop:
            texts, labels = texts.to(device), labels.to(device)                                 # Move batch to specified device
            optimizer.zero_grad()                                                               # Zero gradients
            outputs = model(texts)                                                              # Forward pass
            cost = cost_function(outputs, labels)                                               # Compute loss
            cost.backward()                                                                     # Backpropagate gradients
            optimizer.step()                                                                    # Update parameters to minimize cost
            total_cost += cost.item()                                                           # Accumulate cost
            loop.set_postfix(cost = cost.item())                                                # Show current batch loss in progress bar
        print(f"Average cost: {total_cost / len(train_loader):.4f}")

    if save_path:                                                                               # Checks for a save path for saving vocab 
        torch.save(model.state_dict(), save_path)                                               # Saves model's parameters (weights and bias) to given path
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")                                                           # Use GPU if available
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")                                                            # Use MPS if CUDA not available
    else:
        device = torch.device("cpu")                                                            # Otherwise, CPU
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))                                       # Ensures file paths are relative to script location instead of terminal's directory 
    path = os.path.join(base_dir, "imdb_data", "IMDB Dataset.csv") 
    df = pd.read_csv(path)
    dataset = IMDBDataset(df)

    train_size = int(0.8 * len(df))                                                             # 80% of Dataset for training, 20% for testing
    test_size = len(df) - train_size
    train_dataset, _ = random_split(dataset, [train_size, test_size])                           # Randomly split into training and test sets
    train_loader = DataLoader(
        train_dataset, 
        batch_size = 32, 
        shuffle = True,
        collate_fn = collate_fn
        )

    model = LSTMSentiment(len(dataset.vocab)).to(device)
    os.makedirs("models", exist_ok = True) # 
    train_model(model, 
                train_loader, 
                device, 
                epochs = 5, 
                save_path = MODEL_PATH
                )