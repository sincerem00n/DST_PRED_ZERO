import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(1), self.hidden_layer_size).to(self.device)
        c0 = torch.zeros(1, input_seq.size(1), self.hidden_layer_size).to(self.device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))

        predictions = self.linear(lstm_out)
        return predictions