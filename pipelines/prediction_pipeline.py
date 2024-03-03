from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from src.logging.logging import logging
from src.exceptions.exceptions import CustomException
import sys
from src.components.model_building import ResNet11
from src.uitls.uitls import get_default_device
from src.components.prediction import ModelPredictor

class PredictionPipeline:
    def __init__(self, model_path="/workspaces/Image_classification-with-ResNets-/artifacts/resnet11.pt"):
        try:
            self.device = get_default_device()
            self.model = ResNet11()
            self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))  # Load the model
            self.predictor = ModelPredictor(self.model)
        except Exception as e:
            logging.error("Error occurred during pipeline initialization")
            raise CustomException(e, sys)

    def predict(self, input_data):
        try:
            item,confidence = self.predictor.predict(input_data)
            return item,confidence
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    try:
        image = Image.open("/workspaces/Image_classification-with-ResNets-/data/cifar10/test/cat/0990.png")

        # Define the transformation
        transform = transforms.ToTensor()

        # Apply the transformation to convert the image to a tensor
        image_tensor = transform(image)

        # Assume input_data is the data to be predicted  # Example input data
        model_path = '/workspaces/Image_classification-with-ResNets-/artifacts/resnet11.pt' 

        prediction_pipeline = PredictionPipeline(model_path)
        probs,predictions = prediction_pipeline.predict(image_tensor)
        print("Predictions:", predictions,"with prob of ",probs)
    except CustomException as e:
        print("Custom Exception occurred:", e)
