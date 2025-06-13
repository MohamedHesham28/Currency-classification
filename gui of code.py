import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from model import CurrencyClassifier, get_device
from utils import preprocess_image, load_model, get_class_names
import os

class CurrencyClassifierGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("US Currency Classifier")
        self.root.geometry("800x600")
        
        # Initialize model
        self.device = get_device()
        self.model = CurrencyClassifier().to(self.device)
        self.class_names = get_class_names('USA currency data')
        
        # Load trained model if exists
        if os.path.exists('currency_classifier.pt'):
            self.model = load_model(self.model, 'currency_classifier.pt')
            self.model.eval()
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Upload button
        self.upload_btn = tk.Button(
            self.root,
            text="Upload Currency Image",
            command=self.upload_image,
            font=('Arial', 12)
        )
        self.upload_btn.pack(pady=20)
        
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Prediction label
        self.prediction_label = tk.Label(
            self.root,
            text="",
            font=('Arial', 14)
        )
        self.prediction_label.pack(pady=20)
        
        # Confidence label
        self.confidence_label = tk.Label(
            self.root,
            text="",
            font=('Arial', 12)
        )
        self.confidence_label.pack(pady=10)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Display image
                image = Image.open(file_path)
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                
                # Make prediction
                self.predict_currency(file_path)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def predict_currency(self, image_path):
        try:
            # Preprocess image
            image = preprocess_image(image_path)
            image = image.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Update labels
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item() * 100
            
            self.prediction_label.config(
                text=f"Predicted Currency: {predicted_class}"
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence_score:.2f}%"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = CurrencyClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 