import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import welch, butter, filtfilt

# Signal processing functions
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def extract_features(data, fs):
    features = []
    for row in data:
        # Time-domain features
        mean = np.mean(row)
        std = np.std(row)
        skewness = np.mean((row - mean)**3) / (std**3)
        kurtosis = np.mean((row - mean)**4) / (std**4)
        
        # Frequency-domain features using Welch's method
        freqs, psd = welch(row, fs)
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta_power = np.sum(psd[(freqs >= 13) & (freqs < 30)])
        gamma_power = np.sum(psd[(freqs >= 30)])
        
        # Wavelet transform features
        coeffs = pywt.wavedec(row, 'db4', level=5)
        wavelet_features = np.hstack([np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs])
        
        feature_vector = [mean, std, skewness, kurtosis, delta_power, theta_power, alpha_power, beta_power, gamma_power]
        feature_vector.extend(wavelet_features)
        features.append(feature_vector)
        
    
    return np.array(features)

# Neural network model
class CNNLSTMTransformerModel(nn.Module):
    def __init__(self, input_shape, num_heads=8, ff_dim=128):
        super(CNNLSTMTransformerModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        
        # Transformer layers
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(64)
        self.layernorm2 = nn.LayerNorm(64)
        self.ff = nn.Sequential(
            nn.Linear(64, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 64)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.layernorm2(x + ff_output)
        
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x

# Training and evaluation functions
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader)}')

# Assuming data is loaded and preprocessed
# X_train, y_train, X_test, y_test = your_data_loader_function()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # type: ignore
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate and train the model
model = CNNLSTMTransformerModel(input_shape=(X_train_tensor.shape[1], 1))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=50)
evaluate_model(model, test_loader, criterion)
