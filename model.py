import torch
import torch.nn as nn

class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim = 128, hidden_dim = 128, output_dim = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)                                    # Embedding layer converts token IDs to embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)                          # LSTM layer with batch dimension first
        self.fc = nn.Linear(hidden_dim, output_dim)                                             # Fully connected layer for final prediction

    def forward(self, x):
        x = self.embedding(x)                                                                   # Convert input IDs to embeddings
        _, (fhs, _) = self.lstm(x)                                                              # Pass embeddings through LSTM; take final hidden state of last time step
        out = self.fc(fhs[-1])                                                                  # Map final idden state to output logits
        return out