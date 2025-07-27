# Import libraries
import os, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchinfo import summary
from training import get_mean_std, train, predict, train_callbacks, early_stopping

# Set device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load and preprocess the dataset
train_dir = os.path.join('potato_dataset', 'train')
classes = os.listdir(train_dir)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=train_dir, transform=transform)
mean, std = get_mean_std(DataLoader(dataset, batch_size=32))
transform_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
norm_dataset = datasets.ImageFolder(root=train_dir, transform=transform_norm)
train_dataset, val_dataset = random_split(norm_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define a custom CNN model
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
    nn.Flatten(), nn.Dropout(),
    nn.Linear(64 * 3 * 3, 500), nn.ReLU(), nn.Dropout(),
    nn.Linear(500, 3)
)
model.to(device)

# Train the custom CNN
loss_fn, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters())
train(model, optimizer, loss_fn, train_loader, val_loader, epochs=15, device=device)

# Transfer learning using ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for params in model.parameters():
    params.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 3)
)
model.to(device)

# Train the ResNet model
loss_fn, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters())
train(model, optimizer, loss_fn, train_loader, val_loader, epochs=10, device=device)

# Test prediction
test_dataset = datasets.ImageFolder(root=os.path.join("potato_dataset", "test"), transform=transform_norm)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
test_probabilities = predict(model, test_loader, device)
test_predictions = torch.argmax(test_probabilities, dim=1)

# Train with early stopping and save best model
train_callbacks(
    model, optimizer, loss_fn, train_loader, val_loader,
    epochs=50, device=device,
    checkpoint_path="LR_model.pth", early_stopping=early_stopping
)

# Load the best model and predict
model.load_state_dict(torch.load("LR_model.pth")["model_state_dict"])
test_probabilities = predict(model, test_loader, device)
test_predictions = torch.argmax(test_probabilities, dim=1)
