import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
import os
from model import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description='Add these argument for training')
parser.add_argument('--dir', default='results', help='directory for saving trianed mode')
# parser.add_argument('--feature', default='Scalar_B,BX_GSE_GSM,BY_GSE,BZ_GSE,BY_GSM,BZ_GSM,Proton_Density,SW_Plasma_Temperature,SW_Plasma_Speed', help='features for training')

parser.add_argument('--lr', default=0.0005, help='learning rate')
parser.add_argument('--epochs', default=100, help='epoch number')
parser.add_argument('--batch_size', default=32)
args = parser.parse_args()

## Set parameter for the model
directory = args.dir                # directory for saving model
batch_size = int(args.batch_size)   # batch size
learning_rate = float(args.lr)
num_epochs = int(args.epochs)

data = pd.read_csv('data/solar_wind_parameters_data_1_hourly_all.csv')

data = data.drop(columns=['Unnamed: 0','Timestamp'])

# data.head()

features = ['Scalar_B', 'BX_GSE_GSM', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM', 'Proton_Density', 'SW_Plasma_Temperature', 'SW_Plasma_Speed']
target = ['Dst-index']

# Select the features and target
data = data[features + target]

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into sequences
def sequence(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1]) # all columns except the last
        y.append(data[i+seq_length, -1])  # target is the last column
        # print(f'X:{np.array(X)}, Y:{np.array(y)}')
    return np.array(X), np.array(y)

SEQ_LENGTH = 24
X, y = sequence(data_scaled, SEQ_LENGTH)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=False)
print(f'Train shape: {X_train.shape}, {y_train.shape}')
print(f'Test shape: {X_test.shape}, {y_test.shape}')

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_load = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_load = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# --- Training the model ---

if __name__ == "__main__":
    input_size = len(features)
    hidden_layer_size = 64
    output_size = 1
    model = LSTM(input_size, hidden_layer_size, output_size).to(device)
    print(f'Model: {model} input_size: {input_size}, hidden_layer_size: {hidden_layer_size}, output_size: {output_size}')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    train_loss = 0.0
    best_loss = 1e10
    best_epoch = 0
    rec = []
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        for i, data in enumerate(train_load):
            X_train, y_train = data
            optimizer.zero_grad()
            y_pred = model(X_train)

            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        end = time.time()
            
        train_loss = train_loss / len(train_load)

        # Test the model
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_load):
                X_test, y_test = data
                y_pred = model(X_test)
                loss = criterion(y_pred, y_test)
                test_loss += loss.item()
        test_loss = test_loss / len(test_load)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {end-start:.2f} sec")
    
        if test_loss < best_loss:
            print(f"Model improved from {best_loss:.4f} to {test_loss:.4f} Saving model...")
            best_loss = test_loss
            best_epoch = epoch + 1

            # Save checkpoint
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }
            torch.save(checkpoint, f'./{directory}/model{round(best_loss*10000)}.pth')
            torch.save(checkpoint, f'./{directory}/best_model.pth')

        # Record results
        rec.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
        })

    # Write results to csv
    df = pd.DataFrame(rec)
    df.to_csv(f'{directory}/results.csv', index=False)

    print(f"Finihsed training after {num_epochs} epochs. Best loss: {best_loss:.4f} at epoch {best_epoch}")
    
