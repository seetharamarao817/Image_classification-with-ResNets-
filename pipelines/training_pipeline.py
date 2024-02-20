import torch
from src.logger import logging
from src.exception import CustomException
import sys

class TrainingPipeline:
    def __init__(self, data_dir, batch_size, num_workers=3, pin_memory=True):
        try:
            self.data_loader = CIFAR10DataHandler(data_dir, batch_size, num_workers, pin_memory)
            self.device = get_default_device()
            self.model = to_device(ResNet11(in_channels=3, num_classes=10),self.device)
        except Exception as e:
            logging.error("Error occurred during initialization of TrainingPipeline")
            raise CustomException(e, sys)

    def train(self, epochs, max_lr, opt_func=torch.optim.SGD,
              weight_decay=0, grad_clip=None):
        try:
            train_dl, valid_dl = self.data_loader.get_data_loaders()

            train_dl = DeviceDataLoader(train_dl, self.device)
            valid_dl = DeviceDataLoader(valid_dl, self.device)

            history = fit_one_cycle(epochs, max_lr, self.model, train_dl, valid_dl,
                                    weight_decay, grad_clip, opt_func)
            torch.save(self.model.state_dict(),'./resnet11.pt')
            return history
        except Exception as e:
            logging.error("Error occurred during training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        data_dir = './data/cifar10'
        batch_size = 64
        num_workers = 3
        pin_memory = True

        training_pipeline = TrainingPipeline(data_dir, batch_size, num_workers, pin_memory)
        history = training_pipeline.train(epochs=10, max_lr=0.01)
    except CustomException as e:
        print("Custom Exception occurred:", e)
