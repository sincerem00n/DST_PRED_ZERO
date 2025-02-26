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
        
        # CNN layer + Max Pooling
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=32, 
            kernel_size=3, 
            padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.lstm = nn.LSTM(32, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, features, time)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Convert back to (batch, time, features)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        out = self.fc(x)
        return out
    
class DSTNET(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_size, num_layers=num_layers, batch_first=True)

       # Dropout layers
        self.dropout = nn.Dropout(0.4)
        
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

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Increased regularization with dropout
        self.dropout = nn.Dropout(0.5)
        
        # CNN layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Bidirectional LSTM with dropout
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers, 
                           batch_first=True, 
                           dropout=0.3,
                           bidirectional=True)
        
        # Output layers with regularization
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN layers
        x = x.permute(0, 2, 1)
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        
        # LSTM layer
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep and apply dense layers
        x = lstm_out[:, -1, :]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.bn3(x)
        out = self.fc2(x)
        return out

class CNN_LSTM_AR(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(32, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

        # Add an embedding layer to project 1D predictions back to input dimension
        self.embedding = nn.Linear(output_size, input_size)

    def forward(self, x, future_steps=1):
        """
        x: Input sequence (batch, time_steps, features)
        future_steps: Number of future time steps to predict autoregressively
        """
        batch_size = x.size(0)
        predictions = []

        # Pass input through CNN
        x = x.permute(0, 2, 1)  # Convert to (batch, features, time)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # x = self.pool(x)
        x = x.permute(0, 2, 1)  # Convert back to (batch, time, features)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])  # Get last time step
        pred = self.fc(x)
        predictions.append(pred)

        # Autoregressive Prediction Loop
        for _ in range(future_steps - 1):
            pred_embedded = self.embedding(pred)  # (batch, input_size)
            
            # Prepare for CNN
            pred_input = pred_embedded.unsqueeze(2)  # (batch, input_size, 1)
            
            # CNN pass
            conv_out = torch.relu(self.conv1(pred_input))
            conv_out = torch.relu(self.conv2(conv_out))
            # pooled = self.pool(conv_out)
            
            # LSTM pass
            lstm_in = conv_out.permute(0, 2, 1)  # (batch, time, features)
            lstm_out, (h_n, c_n) = self.lstm(lstm_in, (h_n, c_n))
            
            # Generate new prediction
            x = self.dropout(lstm_out[:, -1, :])
            pred = self.fc(x)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)  # (batch, future_steps, output_size)
