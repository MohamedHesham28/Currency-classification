
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CurrencyClassifier, get_device
from utils import load_data, save_model
import matplotlib.pyplot as plt
import time

def train_model(
    data_dir,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    num_conv_layers=3,
    num_fc_layers=2,
    dropout_rate=0.5,
    model_save_path='currency_classifier.pt'
):
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, _ = load_data(data_dir, batch_size=batch_size)

    # Initialize model
    model = CurrencyClassifier(
        num_conv_layers=num_conv_layers,
        num_fc_layers=num_fc_layers,
        dropout_rate=dropout_rate
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print progress
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 50)

    # Save the model
    save_model(model, model_save_path)
    print(f'Model saved to {model_save_path}')

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    # Example usage
    train_model(
        data_dir='USA currency data',
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001,
        num_conv_layers=3,
        num_fc_layers=2,
        dropout_rate=0.5
    ) 