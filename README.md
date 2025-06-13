# US Currency Classification System

This project implements a Convolutional Neural Network (CNN) for classifying US currency denominations using PyTorch. The system can classify images of US currency notes into six categories: $1, $2, $5, $10, $50, and $100.

## Features

- CNN-based currency classification
- GPU support for faster training
- Adjustable hyperparameters
- Simple GUI for image upload and prediction
- Training visualization
- Model saving and loading

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- tkinter
- matplotlib
- scikit-learn

Install all requirements using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: CNN architecture definition
- `train.py`: Training script
- `gui.py`: Graphical user interface
- `utils.py`: Helper functions
- `requirements.txt`: Project dependencies

## Usage

### Training the Model

1. Place your currency images in the `USA currency data` directory, organized in subdirectories by denomination:
   ```
   USA currency data/
   ├── 1 Dollar/
   ├── 2 Dollar/
   ├── 5 Dollar/
   ├── 10 Dollar/
   ├── 50 Dollar/
   └── 100 Dollar/
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

   You can adjust hyperparameters in the `train.py` file:
   - Number of epochs
   - Batch size
   - Learning rate
   - Number of convolutional layers
   - Number of fully connected layers
   - Dropout rate

### Using the GUI

1. After training, run the GUI:
   ```bash
   python gui.py
   ```

2. Click "Upload Currency Image" to select an image
3. The system will display the image and show the predicted denomination with confidence score

## Model Architecture

The CNN architecture includes:
- Multiple convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Fully connected layers
- Softmax output layer

## Notes

- The model automatically uses GPU if available
- Training progress is saved in `training_history.png`
- The trained model is saved as `currency_classifier.pt`
- The dataset is automatically split into training (70%), validation (20%), and test (10%) sets 