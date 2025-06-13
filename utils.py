import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

def load_data(data_dir, batch_size=32, img_size=128):
    """
    Load and preprocess the dataset
    Returns train, validation, and test dataloaders
    """
    # Define transformations with fixed size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load dataset
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Create dataloaders with num_workers=0 for Windows compatibility
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

        print(f"Dataset loaded successfully:")
        print(f"Total images: {total_size}")
        print(f"Training images: {train_size}")
        print(f"Validation images: {val_size}")
        print(f"Test images: {test_size}")

        return train_loader, val_loader, test_loader
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def preprocess_image(image_path, img_size=128):
    """
    Preprocess a single image for inference
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

def save_model(model, path):
    """Save the model to a file"""
    try:
        torch.save(model.state_dict(), path)
        print(f"Model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def load_model(model, path):
    """Load the model from a file"""
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def get_class_names(data_dir):
    """Get the class names from the dataset directory"""
    try:
        classes = sorted(os.listdir(data_dir))
        print(f"Found classes: {classes}")
        return classes
    except Exception as e:
        print(f"Error getting class names: {str(e)}")
        raise 
    