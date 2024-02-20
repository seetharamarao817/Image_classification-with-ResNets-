from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from src.logger import logging
from src.exception import CustomException
import sys


class PredictionPipeline:
    def __init__(self, model_path):
        try:
            self.device = get_default_device()
            self.model = ResNet11()
            self.model.load_state_dict(torch.load(model_path))  # Load the model
            self.predictor = ModelPredictor(self.model)
        except Exception as e:
            logging.error("Error occurred during pipeline initialization")
            raise CustomException(e, sys)

    def predict(self, input_data):
        try:
            predictions = self.predictor.predict(input_data)
            return predictions
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    try:
        image = Image.open("/kaggle/working/data/cifar10/test/horse/0487.png")

        # Define the transformation
        transform = transforms.ToTensor()

        # Apply the transformation to convert the image to a tensor
        image_tensor = transform(image)

        # Assume input_data is the data to be predicted  # Example input data
        model_path = '/kaggle/working/resnet11.pt'  # Example model path

        prediction_pipeline = PredictionPipeline(model_path)
        probs,predictions = prediction_pipeline.predict(image_tensor)
        print("Predictions:", predictions,"with prob of ",probs)
    except CustomException as e:
        print("Custom Exception occurred:", e)
