import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train_autoencoders import Autoencoder

batch_size = 128
latent_dim = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


def classify(autoencoders, data_loader, device):
    correct = 0
    total = 0
    criterion = nn.MSELoss(reduction='none')
    original_images = [[] for _ in range(10)]
    reconstructed_images = [[] for _ in range(10)]
    predicted_labels = [-1] * 10
    visualized_classes = set()

    for images, labels in data_loader:
        images = images.view(-1, 28 * 28).to(device)
        reconstruction_errors = torch.zeros(images.size(0), len(autoencoders), device=device)

        for i, autoencoder in enumerate(autoencoders):
            with torch.no_grad():
                reconstructions = autoencoder(images)
                errors = criterion(reconstructions, images).mean(dim=1)
                reconstruction_errors[:, i] = errors

        predicted_labels_batch = torch.argmin(reconstruction_errors, dim=1)
        correct += (predicted_labels_batch == labels.to(device)).sum().item()
        total += labels.size(0)

        for idx, label in enumerate(labels):
            label = label.item()
            if label not in visualized_classes:
                original_images[label].append(images[idx].cpu())
                reconstructed_images[label].append(autoencoders[label](images[idx].unsqueeze(0)).cpu())
                predicted_labels[label] = predicted_labels_batch[idx].item()
                visualized_classes.add(label)

    for label in range(10):
        if label not in visualized_classes:
            print(f"No image predicted as class {label}. Adding placeholder.")
            original_images[label].append(torch.zeros(28 * 28).view(1, 28, 28))
            reconstructed_images[label].append(torch.zeros(28 * 28).view(1, 28, 28))

    accuracy = correct / total
    return accuracy, original_images, reconstructed_images, predicted_labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoders = []

    for idx in range(10):
        autoencoder = Autoencoder(latent_dim)
        autoencoder.load_state_dict(torch.load(f"autoencoder_class_{idx}.pth", weights_only=True))
        autoencoder.to(device)
        autoencoder.eval()
        autoencoders.append(autoencoder)

    accuracy, original_images, reconstructed_images, predicted_labels = classify(autoencoders, test_loader, device)
    print(f"\nClassification accuracy: {accuracy * 100:.2f}%")


    def visualize_results(original_images, reconstructed_images, predicted_labels):
        fig, axes = plt.subplots(2, 10, figsize=(15, 4))
        for i in range(10):
            if original_images[i]:
                axes[0, i].imshow(original_images[i][0].view(28, 28).detach().numpy(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(reconstructed_images[i][0].view(28, 28).detach().numpy(), cmap='gray')
                axes[1, i].axis('off')
                axes[0, i].set_title(f"Class {i}")
                axes[1, i].imshow(reconstructed_images[i][0].view(28, 28).detach().numpy(), cmap='gray')
                axes[1, i].axis('off')
                axes[1, i].set_title(f"Predicted: {predicted_labels[i]}")
            else:
                axes[0, i].text(0.5, 0.5, "No Image", ha='center', va='center', transform=axes[0, i].transAxes)
                axes[1, i].text(0.5, 0.5, "No Reconstruction", ha='center', va='center', transform=axes[1, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        plt.suptitle("Top: Original Images, Bottom: Reconstructions")
        plt.show()


    visualize_results(original_images, reconstructed_images, predicted_labels)