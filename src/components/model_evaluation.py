import torch
from src.logging.logging import logging
from src.exceptions.exceptions import CustomException
import sys

@torch.no_grad()
def evaluate(model, val_loader):
    try:
        logging.info("Evaluation started")
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        result = model.validation_epoch_end(outputs)
        logging.info("Evaluation completed")
        return result
    except Exception as e:
        logging.error("Error occurred during evaluation")
        raise CustomException(e, sys)
