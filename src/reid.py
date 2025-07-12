import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# --- Model and Preprocessing Setup ---
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet-18 model
model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
# We use the model up to the penultimate layer to get feature embeddings
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
model.to(device)

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
        input_batch = input_tensor.unsqueeze(0)
        input_batch.to(device)

        # Get the feature embedding
        with torch.no_grad():
            embedding = model(input_batch)

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
