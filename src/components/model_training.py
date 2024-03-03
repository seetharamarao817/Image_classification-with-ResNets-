import torch
import sys
import torch.nn as nn
from src.logging.logging import logging
from src.exceptions.exceptions import CustomException
from src.components.model_evaluation import evaluate  

def get_lr(optimizer):
    try:
        for param_group in optimizer.param_groups:
            return param_group['lr']
    except Exception as e:
        logging.error("Error occurred while getting learning rate")
        raise CustomException(e, sys)

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    try:
        torch.cuda.empty_cache()
        history = []

        # Set up custom optimizer with weight decay
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()

            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
        return history
    except Exception as e:
        logging.error("Error occurred during training")
        raise CustomException(e, sys)
