import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import IMDBDataset, collate_fn
from model import LSTMSentiment
from train import MODEL_PATH
import os


def evaluate_model(model, test_loader, device):
    model.eval()                                                                                # Set model to evaluation mode (no dropout or batch norm)
    correct, total = 0, 0
    with torch.no_grad():                                                                       # Disable gradient computation
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)                                                              # Get model output (logits)
            predictions = torch.argmax(outputs, dim = 1)                                        # Get predicted class index for output
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total

    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    return accuracy

def visualize_predictions(model, dataset, device, n = 5):                                       # Visualize predictions during testing
    model.eval()
    with torch.no_grad():
        for i in range(n):
            token_ids, true_label = dataset[i]                                                  # Get token tensor and true label of given index
            token_ids = token_ids.unsqueeze(0).to(device)                                       # Add batch dimension and move to specified device
            output = model(token_ids)                                                           # Get model output (logits)
            pred_label = torch.argmax(output, dim = 1).item()                                   # Get predicted class index of output
            sentiment_map = {0: "negative", 1: "positive"}                                      # Map output to sentiment

            print(f"Text: {dataset.texts[i][:100]}...")
            print(f"True: {sentiment_map[true_label.item()]}, Pred: {sentiment_map[pred_label]}\n")


def predict_text(model, vocab, text, device, max_length = 256):
    """
    Takes a raw string input, encodes it using the same vocab,
    and returns the predicted sentiment.
    """
    model.eval()
    tokens = text.lower().split()                                                               # Convert text to tokens by space
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]                          # Looks up token ID in vocab, if not there replace with unknown token (unk)
    token_ids = token_ids[:max_length]                                                          # Truncates token sequence if exceeds max length. Ensures sequences aren't too long for the model
    token_tensor = torch.tensor(token_ids, dtype = torch.long).unsqueeze(0).to(device)          # Convert list of ints to tensor (type long)

    with torch.no_grad():
        output = model(token_tensor)
        pred_label = torch.argmax(output, dim = 1).item()                                       # selects index of the largest logit along output_dimension

    sentiment_map = {0: "negative", 1: "positive"}
    return sentiment_map[pred_label]


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")                                                           # Use GPU if available
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")                                                            # Use MPS if CUDA not available
    else:
        device = torch.device("cpu")                                                            # Otherwise, CPU

    base_dir = os.path.dirname(os.path.abspath(__file__))                                       # Ensures file paths are relative to script location instead of terminal's directory
    path = os.path.join(base_dir, "imdb_data", "IMDB Dataset.csv")
    df = pd.read_csv(path)
    dataset = IMDBDataset(df)
    
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, 
                             batch_size = 32, 
                             shuffle = True, 
                             collate_fn = collate_fn
                             )

    model = LSTMSentiment(len(dataset.vocab)).to(device)                                        # Initialize LSTM model, move to specified device
    model.load_state_dict(torch.load(MODEL_PATH, map_location = device))

    print("Model loaded")
    evaluate_model(model, test_loader, device)
    visualize_predictions(model, dataset, device)

    print("\nSentiment Prediction")
    print("Type a review or 'quit' to exit:")
    while True:
        user_input = input("Enter a review: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Exiting interactive mode.")
            break
        sentiment = predict_text(model, dataset.vocab, user_input, device)
        print(f"Predicted sentiment: {sentiment}\n")