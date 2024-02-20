


class ModelPredictor:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def predict(self, input_data):
        try:
            device = get_default_device()
            xb = input_data.unsqueeze(0)
            to_device(xb, device)
            outputs = self.model(xb)
            logging.info("Predictions generated successfully")
            probs,preds = torch.max(outputs, dim=1)
            return probs,preds
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)
