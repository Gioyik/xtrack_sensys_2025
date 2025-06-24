import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

# --- Model and Preprocessing Setup ---
# Load a pre-trained ResNet-18 model
model = resnet18(pretrained=True)
# We use the model up to the penultimate layer to get feature embeddings
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

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
    if person_image.size == 0:
        return None

    # Convert numpy image (from OpenCV) to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

    # Preprocess the image and add a batch dimension
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    # Use GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    # Get the feature embedding
    with torch.no_grad():
        embedding = model(input_batch)

    # Flatten the embedding and convert to a numpy array
    return embedding.squeeze().cpu().numpy()


def cosine_similarity(embedding1, embedding2):
    """
    Calculates the cosine similarity between two embeddings.
    """
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
