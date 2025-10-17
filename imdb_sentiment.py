import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm.auto import tqdm


class IMDBDataset(Dataset):                                                                     
    def __init__(self, df, vocab=None, max_length=256):
        self.texts = df['review'].values                                                        # Extract the review text column as a NumPy array
        self.labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values                # Map 'positive' to 1 and 'negative' to 0
        self.max_length = max_length                                                              # Maximum token length for truncation/padding

        if vocab is None:
            self.vocab = {"<PAD>": 0, "<UNK>": 1}                                               # Initialize vocab with PAD and UNK tokens
            idx = 2
            for text in self.texts:
                for token in text.lower().split():                                               # Split text into tokens
                    if token not in self.vocab:                                                 # If token is new, add to vocab
                        self.vocab[token] = idx
                        idx += 1
        else:
            self.vocab = vocab                                                                    # Use provided vocab if given

    def __len__(self):
        return len(self.texts)                                                                    # Total number of reviews
    
    def encode_text(self, text):
        tokens = text.lower().split()                                                            # Tokenize text by spaces
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]             # Convert tokens to indices, use <UNK> if not in vocab
        token_ids = token_ids[:self.max_length]                                                  # Truncate to max_length
        return torch.tensor(token_ids, dtype=torch.long)                                         # Return as a PyTorch tensor

    def __getitem__(self, idx):
        token_ids = self.encode_text(self.texts[idx])                                            # Encode the text at index idx
        label = torch.tensor(self.labels[idx], dtype=torch.long)                                 # Convert label to tensor
        return token_ids, label                                                                  # Return token tensor and label tensor
    

class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)                                     # Embedding layer converts token IDs to embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)                             # LSTM layer with batch dimension first
        self.fc = nn.Linear(hidden_dim, output_dim)                                              # Fully connected layer for final prediction

    def forward(self, x):
        x = self.embedding(x)                                                                    # Convert input IDs to embeddings
        _, (h_n, _) = self.lstm(x)                                                              # Pass embeddings through LSTM; take hidden state of last time step
        out = self.fc(h_n[-1])                                                                  # Map hidden state to output logits
        return out
    

def collate_fn(batch):
    texts, labels = zip(*batch)                                                                 # Separate token tensors and labels from batch
    texts_padded = pad_sequence(
        texts, 
        batch_first=True, 
        padding_value=0                                                                         # Pad sequences to same length with <PAD> token
    )
    labels = torch.stack(labels)                                                                # Stack labels into a tensor
    return texts_padded, labels                                                                 # Return padded texts and labels
    

def visualize_predictions(model, dataset, device, n=10):
    model.eval()                                                                                 # Set model to evaluation mode (no dropout, batch norm)
    with torch.no_grad():                                                                        # Disable gradient computation
        for i in range(n):
            token_ids, true_label = dataset[i]                                                  # Get token tensor and true label
            token_ids = token_ids.unsqueeze(0).to(device)                                       # Add batch dimension and move to device
            output = model(token_ids)                                                           # Get model output (logits)
            pred_label = torch.argmax(output, dim=1).item()                                     # Predicted class index
            
            sentiment_map = {0: "negative", 1: "positive"}                                      # Map index to sentiment
            print(f"Text: {dataset.texts[i][:100]}...")                                         # Print first 100 chars of text
            print(f"True: {sentiment_map[true_label.item()]}, Pred: {sentiment_map[pred_label]}\n")
    model.train()                                                                                # Return model to training mode


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")                                                           # Use GPU if available
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")                                                            # Use MPS if CUDA not available
    else:
        device = torch.device("cpu")                                                            # Otherwise, CPU
    print(f"Using device: {device}")

    path = "imdb_data/IMDB Dataset.csv"
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)                      # Load IMDB dataset

    full_dataset = IMDBDataset(df, max_length=256)                                               # Initialize dataset
    train_size = int(0.8 * len(df))                                                             
    test_size = len(df) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])           # Randomly split into train and test sets

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn)                                                                  # DataLoader for training with padding
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn)                                                                  # DataLoader for testing

    model = LSTMSentiment(
        vocab_size=len(full_dataset.vocab), 
        embed_dim=128, 
        hidden_dim=128, 
        output_dim=2)                                                                            # Initialize LSTM model
    model.to(device)                                                                             # Move model to device

    cost_function = nn.CrossEntropyLoss()                                                        # Cross-entropy loss for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                                   # Adam optimizer for parameter updates

    epochs = 5
    for epoch in range(epochs):
        model.train()                                                                           # Set model to training mode
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')                                       # Progress bar for each epoch
        total_cost = 0
        for texts, labels in loop:
            texts, labels = texts.to(device), labels.to(device)                                  # Move batch to device
            optimizer.zero_grad()                                                               # Zero gradients
            outputs = model(texts)                                                              # Forward pass
            cost = cost_function(outputs, labels)                                               # Compute loss
            cost.backward()                                                                     # Backpropagate gradients
            optimizer.step()                                                                    # Update parameters
            total_cost += cost.item()                                                           # Accumulate loss
            loop.set_postfix(cost=cost.item())                                                  # Show current batch loss in progress bar
        print(f"Average cost: {total_cost/len(train_loader):.4f}")                               # Print epoch loss

    model.eval()                                                                                 # Evaluation mode
    correct, total = 0, 0
    with torch.no_grad():                                                                        # Disable gradients for evaluation
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predictions = torch.argmax(outputs, dim=1)                                          # Get predicted classes
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Validation accuracy: {accuracy * 100:.2f}%")                                        # Print accuracy on test set

visualize_predictions(model, full_dataset, device, n=10)