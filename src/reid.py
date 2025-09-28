import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# Use best available device (CUDA > MPS > CPU)
def get_best_device():
    """
    Automatically select the best available compute device.
    
    Returns:
        torch.device: Best available device
        
    Priority Order:
        1. CUDA (NVIDIA GPU) - if available
        2. MPS (Apple Silicon GPU) - if available  
        3. CPU (fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def validate_device(device_str):
    """
    Validate and return a torch device, falling back if necessary.
    
    Args:
        device_str (str): Device string to validate ("cpu", "cuda", "mps")
        
    Returns:
        torch.device: Validated device
        
    Validation Process:
        1. Check device availability
        2. Fall back to best available device if requested device unavailable
        3. Print warnings for fallbacks
    """
    try:
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to best available device")
            return get_best_device()
        elif device.type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print(f"Warning: MPS not available, falling back to best available device")
            return get_best_device()
        return device
    except Exception:
        print(f"Warning: Invalid device '{device_str}', falling back to best available device")
        return get_best_device()

# Global model and device - will be initialized when needed
_model = None
_device = None

def initialize_reid_model(device_str="auto"):
    """Initialize the ReID model with the specified device"""
    global _model, _device
    
    if device_str == "auto":
        _device = get_best_device()
    else:
        _device = validate_device(device_str)
    
    print(f"Initializing ReID model on device: {_device}")
    
    # Load a pre-trained ResNet-18 model
    _model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
    # We use the model up to the penultimate layer to get feature embeddings
    _model = torch.nn.Sequential(*(list(_model.children())[:-1]))
    _model.eval()
    _model.to(_device)
    
    return _model, _device

# Define the image transformations
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_appearance_embedding(person_image):
    """
    Calculates an appearance embedding for a given person image.
    """
    global _model, _device
    
    # Initialize model if not already done
    if _model is None:
        initialize_reid_model("auto")
    
    try:
        if person_image is None or person_image.size == 0:
            return None

        # Validate image dimensions
        if len(person_image.shape) != 3 or person_image.shape[2] != 3:
            return None

        # Convert numpy image (from OpenCV) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

        # Preprocess the image and add a batch dimension
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(_device)

        # Get the feature embedding
        with torch.no_grad():
            embedding = _model(input_batch)

        # Flatten the embedding and convert to a numpy array
        return embedding.squeeze().cpu().numpy()

    except Exception as e:
        print(f"Error processing image for ReID: {e}.")
        return None


def cosine_similarity(embedding1, embedding2):
    """
    Calculates the cosine similarity between two embeddings.
    """
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
