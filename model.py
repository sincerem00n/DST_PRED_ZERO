import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Final prediction layer

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use last time step output
        return out

    
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, time)
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Convert back to (batch, time, channels)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
    
class DSTNET(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_size, num_layers=num_layers, batch_first=True)

       # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.4, batch_first=True)
        
        # Dense layers with regularization
        self.dense1 = nn.Linear(hidden_size, 400)
        self.dense2 = nn.Linear(400, 100)
        self.dense3 = nn.Linear(100, output_size)
        
        # L1 and L2 regularization will be applied during training using weight_decay in optimizer
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, time)
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Convert back to (batch, time, channels)
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        x = self.dropout(lstm_out)
        
        # Multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.dropout(attn_output)
        
        # Dense layers with dropout and activation
        x = self.dropout(torch.relu(self.dense1(x[:, -1, :])))
        x = self.dropout(torch.relu(self.dense2(x)))
        out = self.dense3(x)

        return out
