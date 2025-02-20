import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, lstm_layers=2, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=lstm_layers, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        
        # Initialize hidden state only if not provided (to allow stateful training)
        if hidden_state is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_layer_size).to(self.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_layer_size).to(self.device)
        else:
            h0, c0 = hidden_state

        lstm_out, hidden_state = self.lstm(input_seq, (h0, c0))

        predictions = self.linear(lstm_out)
        return predictions