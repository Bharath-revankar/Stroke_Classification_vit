import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_image_transforms():
    """
    Returns a dictionary of transformations for training and validation.
    """
    # Normalization parameters are standard for models pre-trained on ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
    }
    return data_transforms

def preprocess_image(image_path):
    """
    Loads an image, resizes it, and applies validation transformations.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    transform = get_image_transforms()['val']
    
    return transform(pil_image)

if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing
    dummy_image_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_image_array)
    dummy_image.save("dummy_image.png")
    
    try:
        preprocessed_tensor = preprocess_image("dummy_image.png")
        print("Shape of preprocessed tensor:", preprocessed_tensor.shape)
        assert preprocessed_tensor.shape == (3, 224, 224)
        print("Preprocessing test passed.")
    except FileNotFoundError as e:
        print(e)
    finally:
        import os
        os.remove("dummy_image.png")
