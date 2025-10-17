import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class IMDBDataset(Dataset):                                                                     
    def __init__(self, df, vocab = None, max_length = 256):
        self.texts = df['review'].values                                                        # Extract the review text column as a NumPy array
        self.labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values                # Map 'positive' to 1 and 'negative' to 0
        self.max_length = max_length                                                            # Maximum token length for truncation/padding

        if vocab is None:
            self.vocab = {"<PAD>": 0, "<UNK>": 1}                                               # Initialize vocab with PAD and UNK tokens
            idx = 2
            for text in self.texts:
                for token in text.lower().split():                                              # Split text into tokens
                    if token not in self.vocab:                                                 # If token is new, add to vocab
                        self.vocab[token] = idx
                        idx += 1
        else:
            self.vocab = vocab                                                                  # Use provided vocab if given

    def __len__(self):
        return len(self.texts)                                                                  # Total number of reviews
    
    def encode_text(self, text):
        tokens = text.lower().split()                                                           # Tokenize text by spaces
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]            # Convert tokens to indices, use <UNK> if not in vocab
        token_ids = token_ids[:self.max_length]                                                 # Truncate to max_length
        return torch.tensor(token_ids, dtype = torch.long)                                      # Return as a PyTorch tensor

    def __getitem__(self, idx):
        token_ids = self.encode_text(self.texts[idx])                                           # Encode the text at index idx
        label = torch.tensor(self.labels[idx], dtype = torch.long)                              # Convert label to tensor
        return token_ids, label                                                                 # Return token tensor and label tensor
    
def collate_fn(batch):
    texts, labels = zip(*batch)                                                                 # Separate token tensors and labels from batch
    texts_padded = pad_sequence(
        texts, 
        batch_first = True, 
        padding_value = 0                                                                       # Pad sequences to same length with <PAD> token
    )
    labels = torch.stack(labels)                                                                # Stack labels into a tensor
    return texts_padded, labels                                                                 # Return padded texts and labels