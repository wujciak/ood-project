import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.config import CONFIG


def train_model(model, loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0.0
        for inputs, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {avg_loss:.4f}")

    return avg_loss
