from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2


class VestClassifier:
    def __init__(self, model_path="vest_model.pth", device="cpu"):
        """
        Initializes the VestClassifier.

        Args:
            model_path (str): Path to the trained PyTorch model file.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
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
        Predicts if a person is wearing a vest using the loaded model.

        Args:
            person_image: The cropped BGR image of the person from OpenCV.

        Returns:
            A tuple of (is_vest, confidence_score).
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
