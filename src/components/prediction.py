import torch
from src.uitls.uitls import get_default_device,to_device

class ModelPredictor:
    def __init__(self, model):
      self.model = model

    @torch.no_grad()
    def predict(self, input_data):
      device = get_default_device()
      xb = input_data.unsqueeze(0)
      to_device(xb, device)
      outputs = self.model(xb)
      _, predicted = torch.max(outputs, 1)
      confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
      return predicted.item(), confidence[predicted].item()
