from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2


class VestClassifier:
    """
    Model-based safety vest detection using PyTorch and MobileNetV2.
    
    This class provides binary classification for detecting whether a person
    is wearing a safety vest using a pre-trained MobileNetV2 model.
    
    Attributes:
        device (torch.device): Compute device for model inference
        model (torch.nn.Module): MobileNetV2 model for binary classification
        transform (torchvision.transforms.Compose): Image preprocessing pipeline
    """
    
    def __init__(self, model_path="vest_model.pth", device="cpu"):
        """
        Initialize the VestClassifier with model path and device.
        
        Args:
            model_path (str): Path to trained PyTorch model file (default: "vest_model.pth")
            device (str): Compute device ("cpu", "cuda", or "mps")
            
        Initialization Process:
            1. Validate and set compute device
            2. Load MobileNetV2 model with custom classifier
            3. Load trained weights if available
            4. Set model to evaluation mode
            5. Move model to specified device
            6. Define image preprocessing pipeline
        """
        self.device = self._validate_device(device)
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Define the image transformations
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    def _validate_device(self, device_str):
        """Validate and return a torch device, falling back if necessary"""
        try:
            device = torch.device(device_str)
            if device.type == "cuda" and not torch.cuda.is_available():
                print(f"Warning: CUDA not available for vest classifier, falling back to CPU")
                return torch.device("cpu")
            elif device.type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print(f"Warning: MPS not available for vest classifier, falling back to CPU")
                return torch.device("cpu")
            return device
        except Exception:
            print(f"Warning: Invalid device '{device_str}' for vest classifier, falling back to CPU")
            return torch.device("cpu")

    def _load_model(self, model_path):
        """Loads the MobileNetV2 model, modified for binary classification."""
        model = mobilenet_v2(weights=None, progress=False)
        # Modify the classifier for 2 classes: (0: no_vest, 1: vest)
        model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

        model_path = Path(model_path)
        if model_path.is_file():
            print(f"Loading trained weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(
                f"Warning: Model file not found at {model_path}. Using un-trained model."
            )
            print("The model will not provide meaningful predictions.")

        return model

    def predict(self, person_image):
        """
        Predict if a person is wearing a safety vest using the loaded model.
        
        Args:
            person_image (numpy.ndarray): Cropped BGR image of person from OpenCV
            
        Returns:
            tuple: (is_vest, confidence_score)
                - is_vest (bool): Whether person is wearing a vest
                - confidence_score (float): Confidence score (0.0 to 1.0)
                
        Process:
            1. Validate input image (non-empty, proper format)
            2. Convert BGR to RGB format
            3. Apply preprocessing transformations
            4. Run model inference
            5. Apply softmax to get probabilities
            6. Return prediction and confidence
        """
        if person_image is None or person_image.size == 0:
            return False, 0.0

        # Convert BGR image to RGB
        rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)

        # Apply transformations and add a batch dimension
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        is_vest = predicted_class.item() == 1
        return is_vest, confidence.item()
