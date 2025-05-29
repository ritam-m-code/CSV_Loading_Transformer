import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ----------------------------
# Transformer Components
# ----------------------------

anomaly_path = Path("4500_1_splitcut_10.csv")
csv_path = Path("8500_3_splitcut_21.csv")
input_window = 50
output_window = 10

def create_dataset_from_csv(path, input_window, output_window, step=10):
    """
    Load CSV and slice it into multiple training samples with given input/output window sizes.
    Args:
        path: Path to CSV file
        input_window: Number of time steps in input sequence
        output_window: Number of time steps in output sequence
        step: Step size for sliding window (default 10)
    Returns:
        X: np.array of shape (num_samples, input_window, num_features)
        Y: np.array of shape (num_samples, output_window, num_features)
    """
    df = pd.read_csv(path)
    features = df.iloc[:, :3].values  # select first 3 columns as features, adjust as needed
    
    total_len = len(features)
    X_list = []
    Y_list = []
    
    for start in range(0, total_len - input_window - output_window + 1, step):
        X_list.append(features[start : start + input_window])
        Y_list.append(features[start + input_window : start + input_window + output_window])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    return X, Y

class PositionalEncoding(nn.Module):
    """Implements transformer positional encoding for temporal sequence data"""
    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """Transformer-based model for multivariate time series forecasting"""
    def __init__(self, num_features=3, input_window=50, output_window=10,
                 d_model=128, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dropout=0.1):
        super().__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.d_model = d_model

        self.input_proj = nn.Linear(num_features, d_model)
        self.decoder_input_proj = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dropout=dropout
        )
        self.output_proj = nn.Linear(d_model, num_features)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def forward(self, src, tgt):
        src = self.input_proj(src)
        src = src.transpose(0, 1)  # (seq_len, batch, features)
        src = self.positional_encoding(src)

        tgt = self.decoder_input_proj(tgt)
        tgt = tgt.transpose(0, 1)
        tgt = self.positional_encoding(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)  # (batch, seq_len, features)
        return self.output_proj(out)

# ----------------------------
# Model Training Implementation
# ----------------------------

def train_model(X_train, Y_train):
    num_features = X_train.shape[2]
    input_window = X_train.shape[1]
    output_window = Y_train.shape[1]
    batch_size = 32
    epochs = 200
    lr = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimeSeriesTransformer(
        num_features=num_features,
        input_window=input_window,
        output_window=output_window,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            # Prepare decoder input: use last input_window timestep from xb as first input to decoder shifted by 1
            # Commonly, decoder input is target shifted by one with initial zero or start token
            decoder_input = torch.zeros_like(yb)
            decoder_input[:, 1:, :] = yb[:, :-1, :]
            
            optimizer.zero_grad()
            output = model(xb, decoder_input)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return model

# ----------------------------
# Main Run
# ----------------------------

if __name__ == "__main__":
    # Load CSV training data
    X_train, Y_train = create_dataset_from_csv(csv_path, input_window, output_window, step=10)
    X_test, Y_test = create_dataset_from_csv(anomaly_path, input_window, output_window, step=10)
    print(f"Loaded training data from CSV: X_train shape {X_train.shape}, Y_train shape {Y_train.shape}")
    
    # Train model
    model = train_model(X_train, Y_train)
    x_test = X_test[0]  # shape: (input_window, num_features)
    y_test = Y_test[0]  # shape: (output_window, num_features)
    # Anomaly testing on synthetic data (you can modify later to use real data)
   # x_test, y_test, t_full, full_sample = generate_anomaly_data(input_window, output_window)
    model.eval()
    
    device = next(model.parameters()).device
    with torch.no_grad():
        src = torch.tensor(x_test[np.newaxis, :, :], dtype=torch.float32).to(device)
        tgt = torch.zeros((1, output_window, x_test.shape[1]), dtype=torch.float32).to(device)  # decoder input zeros
        pred = model(src, tgt)
    
    pred = pred.cpu().numpy()[0]
    
    # Plot the anomaly current channel (index 3)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(input_window), x_test[:, 2], label="Input current")
    plt.plot(np.arange(input_window, input_window + output_window), y_test[:, 2], label="True future current")
    plt.plot(np.arange(input_window, input_window + output_window), pred[:, 2], label="Predicted future current")
    plt.xlabel("Time step")
    plt.ylabel("Current")
    plt.legend()
    plt.title("Transformer Prediction on Anomalous Current Data")
    plt.show()
