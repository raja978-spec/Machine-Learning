# 1. Import Libraries
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from tqdm.notebook import tqdm

# 2. Check Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. Collect Data
train_dir = os.path.join('sea_creatures', 'train')
test_dir = os.path.join('sea_creatures', 'test')

# 4. Transform Data (Resize, Convert to Tensor, Normalize)
class ConvertToRGB:
    def __call__(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

# Initial transform to compute dataset mean & std
basic_transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load initial dataset to compute mean and std
dataset = datasets.ImageFolder(root=train_dir, transform=basic_transform)
loader = DataLoader(dataset, batch_size=32)

# Compute mean & std for normalization
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

mean, std = get_mean_std(loader)

# Final transform with normalization
transform_norm = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Reload dataset with normalized transform
norm_dataset = datasets.ImageFolder(root=train_dir, transform=transform_norm)
train_dataset, val_dataset = random_split(norm_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# Data Loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 5. Define Model (Simple CNN)
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4, 4),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4, 4),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4, 4),
    nn.Flatten(),
    nn.Dropout(), nn.Linear(64 * 3 * 3, 500), nn.ReLU(),
    nn.Dropout(), nn.Linear(500, 9)  # Assuming 9 classes
)
model.to(device)

# 6. Train Model on Train Data
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Basic train function
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=10, device='cpu'):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

train(model, optimizer, loss_fn, train_loader, val_loader, epochs=10, device=device)

# 7. Validate Model Accuracy
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

evaluate(model, val_loader)
