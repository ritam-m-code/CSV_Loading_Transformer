import torch
import torch.nn as nn
import torch.optim as optim
from main import TimeSeriesTransformer  # your model class

# Dummy dataset example (replace with your real DataLoader)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=100, seq_len=50, input_dim=5):
        self.data = torch.randn(n_samples, seq_len, input_dim)
        self.targets = torch.randn(n_samples, 10, input_dim)  # output_window=10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_model():
    model = TimeSeriesTransformer(
        num_features=5,
        input_window=50,
        output_window=10,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.1
    )

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = DummyDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved to trained_model.pth")

train_model()