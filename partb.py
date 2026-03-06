# ============================================
# Assignment 2 - Part B
# Custom Neural Network (Flexible Architecture)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random, os, copy
from typing import List, Tuple, Dict

# ------------------------------
# Set random seed for reproducibility
# ------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ------------------------------
# Flexible Feedforward Network
# ------------------------------
class FlexibleNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', dropout_rate: float = 0.3):
        super().__init__()
        assert len(layer_sizes) >= 4, "Provide at least Input, Hidden1, Hidden2, Output"
        self.activation_name = activation.lower()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))

        self._initialize_weights()

    def _initialize_weights(self):
        for idx, layer in enumerate(self.layers):
            if idx < len(self.layers) - 1:
                if self.activation_name == 'relu':
                    nn.init.kaiming_normal_(layer.weight)
                else:
                    nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.activation_name == 'relu':
                x = torch.relu(x)
            elif self.activation_name == 'tanh':
                x = torch.tanh(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        x = self.layers[-1](x)
        return x


# ------------------------------
# Trainer class
# ------------------------------
class NeuralNetworkTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 batch_size: int = 64, device: str = None):
        self.model = model
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_loss_std': [], 'val_loss_std': []
        }
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def prepare_data(self, X: np.ndarray, y: np.ndarray, val_split: float = 0.2):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        batch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            batch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += batch_X.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / total, 100 * correct / total, batch_losses

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        batch_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                batch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += batch_X.size(0)
                correct += (predicted == batch_y).sum().item()
        return total_loss / total, 100 * correct / total, batch_losses

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, val_split: float = 0.2):
        train_loader, val_loader = self.prepare_data(X, y, val_split)
        print(f"🚀 Training on device: {self.device}")

        for epoch in range(epochs):
            train_loss, train_acc, train_batch_losses = self.train_epoch(train_loader)
            val_loss, val_acc, val_batch_losses = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_loss_std'].append(np.std(train_batch_losses))
            self.history['val_loss_std'].append(np.std(val_batch_losses))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
        return self.history

    def load_best_model(self):
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)


# ------------------------------
# Visualization
# ------------------------------
class PerformanceVisualizer:
    @staticmethod
    def plot_training_curves(history: dict, print_output: bool = True):
        epochs = range(1, len(history['train_loss']) + 1)

        # Optional printing
        if print_output:
            print("\n📊 Training & Validation Metrics per Epoch:")
            print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Train Acc':>10} | {'Val Acc':>10}")
            print("-" * 55)
            for i in range(len(epochs)):
                print(f"{i+1:>5} | {history['train_loss'][i]:>10.4f} | {history['val_loss'][i]:>10.4f} | "
                      f"{history['train_acc'][i]:>10.2f} | {history['val_acc'][i]:>10.2f}")

        # Plotting
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.errorbar(epochs, history['train_loss'], yerr=history['train_loss_std'], label='Train Loss')
        plt.errorbar(epochs, history['val_loss'], yerr=history['val_loss_std'], label='Val Loss')
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'], label='Val Acc')
        plt.title("Training & Validation Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.show()
