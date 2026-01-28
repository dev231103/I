import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# Custom Dataset for Random Sequences
class SequenceDataset(Dataset):
    def __init__(self, num_sequences=1000, sequence_length=10, num_classes=2):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.data = np.random.randn(num_sequences, sequence_length, 1)
        self.labels = np.random.randint(0, num_classes, num_sequences)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# LSTM Model Definition
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # take output from the last time step
        out = self.fc(out)
        return out


# Hyperparameters
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001


# Dataset and DataLoader
dataset = SequenceDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model, Loss, Optimizer
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(num_epochs):
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Test the model with a random sequence
with torch.no_grad():
    test_sequence = torch.randn(1, dataset.sequence_length, input_size).to(device)
    prediction = model(test_sequence)
    predicted_class = torch.argmax(prediction, dim=1).item()
    print(f"Predicted class: {predicted_class}")
