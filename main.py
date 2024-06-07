import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

def load_dataset():
    # Placeholder for loading dataset
    # Replace this with actual data loading logic
    # Example:
    # data = pd.read_csv('path_to_dataset.csv')
    # X = data.drop(columns=['target']).values
    # y = data['target'].values
    X = np.random.randn(100, 1000)  # Example: 100 samples, each with 1000 features
    y = np.random.randn(100)  # Example: 100 target values
    return X, y

def extract_features(X):
    """
    Extract features from the input data.
    
    Parameters:
    X (ndarray): Input data.
    
    Returns:
    ndarray: Extracted features.
    """
    features = []
    for row in X:
        # Time-domain features
        mean = np.mean(row)
        std = np.std(row)
        skewness = np.mean((row - mean)**3) / (std**3)
        kurtosis = np.mean((row - mean)**4) / (std**4)
        
        # Frequency-domain features using Welch's method
        freqs, psd = welch(row)
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
    
    features = np.array(features)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features

class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
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
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=50, patience=5):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
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
        
        scheduler.step(running_loss / len(train_loader))
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

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

# Cross-validation for Gradient Boosting Regressor
def train_gradient_boosting(X, y):
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_val_score(gbr, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f'Gradient Boosting CV MSE: {-np.mean(scores)}')
    gbr.fit(X, y)
    return gbr

# Ensemble prediction
def ensemble_predict(cnn_lstm_model, gbr, X):
    cnn_lstm_model.eval()
    with torch.no_grad():
        cnn_lstm_pred = cnn_lstm_model(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).numpy().flatten()
    gbr_pred = gbr.predict(X)
    final_pred = (cnn_lstm_pred + gbr_pred) / 2
    return final_pred

# Example Usage:

# Load dataset
X, y = load_dataset()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate and train the CNN-LSTM model
cnn_lstm_model = CNNLSTMModel(input_shape=(X_train_tensor.shape[1], 1))
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

train_model(cnn_lstm_model, train_loader, criterion, optimizer, scheduler, epochs=50, patience=5)
evaluate_model(cnn_lstm_model, test_loader, criterion)

# Train Gradient Boosting Regressor with cross-validation
gbr = train_gradient_boosting(X_train_features, y_train)

# Ensemble prediction and evaluation
predictions = ensemble_predict(cnn_lstm_model, gbr, X_test_features)
mse = np.mean((predictions - y_test) ** 2)
mae = np.mean(np.abs(predictions - y_test))
print(f'MSE: {mse}, MAE: {mae}')
