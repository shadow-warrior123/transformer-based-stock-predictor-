import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import TimeSeriesTransformer
from utils import get_device

def train_model(X_train, y_train, input_dim, output_dim=1, seq_length=60, epochs=10, batch_size=32, lr=0.001):
    """
    Trains the Transformer model.
    """
    device = get_device()
    print(f"Training on {device}")
    
    # Create Datasets and Loaders
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    # Initialize Model
    model = TimeSeriesTransformer(input_dim=input_dim, output_dim=output_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.unsqueeze(1)) # Ensure target shape matches
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    return model, loss_history

def predict(model, X_test, scaler):
    """
    Makes predictions using the trained model.
    """
    device = get_device()
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).float().to(device)
        predictions = model(X_tensor)
        predictions = predictions.cpu().numpy()
        
    # We need to inverse transform. The scaler was fitted on (n_samples, n_features).
    # Our predictions are (n_samples, 1) corresponding to the first feature 'Close'.
    # To inverse transform correctly, we need to construct a dummy array with the same shape as input features
    # BUT easier way is if scaler was fitted on just Close. 
    # In preprocessing.py we fitted on ALL cols.
    # So we need to reconstruct the full shape to inverse transform, OR separate scaler for Close.
    # Let's fix this in preprocessing or handle it here.
    # Handling here:
    # Create dummy array
    # scaler.n_features_in_ tells us how many features.
    
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions[:, 0] # Assume Close is at index 0
    
    inversed = scaler.inverse_transform(dummy)[:, 0]
    return inversed
