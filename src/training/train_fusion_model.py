import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from torchvision.datasets import ImageFolder
from PIL import Image

from ..data_preprocessing.image_preprocessing import get_image_transforms
from ..data_preprocessing.clinical_data_preprocessing import preprocess_clinical_data
from ..models.fusion_model import FusionModel

class BimodalFusionDataset(Dataset):
    """
    Custom dataset for loading paired image and clinical data.
    It assumes an ordered correspondence between the image files and clinical data rows.
    """
    def __init__(self, image_folder_path, clinical_csv_path, transform=None):
        self.image_dataset = ImageFolder(image_folder_path, transform=transform)
        self.classes = self.image_dataset.classes
        self.class_to_idx = self.image_dataset.class_to_idx
        
        raw_clinical_data = pd.read_csv(clinical_csv_path)
        
        print("Warning: No direct mapping found between image filenames and clinical data IDs.")
        print("Pairing images and clinical rows by their order (i-th image to i-th clinical row).")
        
        processed_features, processed_labels, self.preprocessor = preprocess_clinical_data(raw_clinical_data.copy(), fit_smote=True)
        
        self.clinical_features = torch.tensor(processed_features, dtype=torch.float32)
        self.clinical_labels = torch.tensor(processed_labels, dtype=torch.long)

        self.num_samples = min(len(self.image_dataset), len(self.clinical_features))
        print(f"Aligned datasets. Using {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, image_label = self.image_dataset[idx]
        clinical_data_row = self.clinical_features[idx]
        
        # Using the image label as the ground truth.
        return image, clinical_data_row, torch.tensor(image_label, dtype=torch.long)

def train_fusion_model():
    """
    Main function to train the bimodal fusion model.
    """
    print("Starting bimodal fusion model training...")

    # --- Configuration ---
    IMAGE_DATA_PATH = 'MRI_DATA/Stroke_classification'
    CLINICAL_DATA_PATH = 'clinical_lab_data/healthcare-dataset-stroke-data.csv'
    IMAGE_MODEL_WEIGHTS = 'models/image_model_weights.pth'
    CLINICAL_MODEL_WEIGHTS = 'models/clinical_model_weights.pth'
    FUSION_MODEL_SAVE_PATH = 'models/fusion_model_weights.pth'
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    if not all(os.path.exists(p) for p in [IMAGE_DATA_PATH, CLINICAL_DATA_PATH, IMAGE_MODEL_WEIGHTS, CLINICAL_MODEL_WEIGHTS]):
        print("Error: Missing necessary data or model weight files.")
        return

    # --- Dataset and DataLoader ---
    print("Loading and pairing datasets...")
    image_transforms = get_image_transforms()
    try:
        fusion_dataset = BimodalFusionDataset(
            image_folder_path=IMAGE_DATA_PATH,
            clinical_csv_path=CLINICAL_DATA_PATH,
            transform=image_transforms['train']
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(fusion_dataset) == 0:
        print("Error: Dataset is empty.")
        return
        
    print(f"Successfully created a dataset with {len(fusion_dataset)} samples.")
    
    train_size = int(0.8 * len(fusion_dataset))
    val_size = len(fusion_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(fusion_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model Initialization ---
    print("Initializing fusion model...")
    clinical_input_size = fusion_dataset.clinical_features.shape[1]
    num_classes = len(fusion_dataset.classes)

    model = FusionModel(
        image_model_weights=IMAGE_MODEL_WEIGHTS,
        clinical_model_weights=CLINICAL_MODEL_WEIGHTS,
        clinical_input_size=clinical_input_size,
        num_classes=num_classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}.")

    # --- Training Setup ---
    optimizer = optim.Adam(model.fusion_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training loop...")
    
    for epoch in range(NUM_EPOCHS):
        model.fusion_head.train()
        model.image_model.eval()
        model.clinical_model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, clinical_data, labels in train_loader:
            images, clinical_data, labels = images.to(device), clinical_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, clinical_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, clinical_data, labels in val_loader:
                images, clinical_data, labels = images.to(device), clinical_data.to(device), labels.to(device)
                outputs = model(images, clinical_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

    # --- Save the Model ---
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), FUSION_MODEL_SAVE_PATH)
    print(f"Bimodal fusion model saved to {FUSION_MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_fusion_model()
