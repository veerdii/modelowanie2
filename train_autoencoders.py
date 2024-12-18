import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

batch_size = 128
latent_dim = 64
learning_rate = 0.001
patience = 1

def get_class_subset(dataset, label):
    indices = [i for i, (_, y) in enumerate(dataset) if y == label]
    return Subset(dataset, indices)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

def train_autoencoder(autoencoder, data_loader, lr, patience, device):
    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, 101):
        total_loss = 0
        for images, _ in data_loader:
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch} - Loss: {average_loss:.4f}")

        if average_loss < best_loss:
            best_loss = average_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
            break

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_subsets = [get_class_subset(train_data, i) for i in range(10)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoders = []
    for label in range(10):
        print(f"\nTraining autoencoder for class {label}...")
        autoencoder = Autoencoder(latent_dim)
        data_loader = DataLoader(train_subsets[label], batch_size=batch_size, shuffle=True)
        train_autoencoder(autoencoder, data_loader, learning_rate, patience, device)
        autoencoders.append(autoencoder)

    # Save autoencoders
    for idx, autoencoder in enumerate(autoencoders):
        torch.save(autoencoder.state_dict(), f"autoencoder_class_{idx}.pth")