import torch.nn.functional as F
import torch.nn as nn
from src.logger import logging
from src.exception import CustomException
import sys

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        try:
            images, labels = batch
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        except Exception as e:
            logging.error("Error occurred during training step")
            raise CustomException(e, sys)

    def validation_step(self, batch):
        try:
            images, labels = batch
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = self.accuracy(out, labels)      # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
        except Exception as e:
            logging.error("Error occurred during validation step")
            raise CustomException(e, sys)

    def validation_epoch_end(self, outputs):
        try:
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        except Exception as e:
            logging.error("Error occurred during validation epoch end")
            raise CustomException(e, sys)

    def epoch_end(self, epoch, result):
        try:
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        except Exception as e:
            logging.error("Error occurred during epoch end")
            raise CustomException(e, sys)

    def accuracy(self, outputs, labels):
        try:
            _, preds = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(preds == labels).item() / len(preds))
        except Exception as e:
            logging.error("Error occurred while calculating accuracy")
            raise CustomException(e, sys)

def conv_block(in_channels, out_channels, pool=False):
    try:
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    except Exception as e:
        logging.error("Error occurred during convolutional block creation")
        raise CustomException(e, sys)

class ResNet11(ImageClassificationBase):
    def __init__(self, in_channels=3, num_classes=10):
        try:
            super().__init__()

            self.conv1 = conv_block(in_channels, 64)
            self.conv2 = conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128),conv_block(128,128))

            self.conv3 = conv_block(128, 256, pool=True)
            self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256),conv_block(256,256))

            self.conv4 = conv_block(256,512,pool=True)
            self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                            nn.Flatten(),
                                            nn.Dropout(0.2),
                                            nn.Linear(512, num_classes))
        except Exception as e:
            logging.error("Error occurred during model initialization")
            raise CustomException(e, sys)

    def forward(self, xb):
        try:
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.res2(out) + out
            out = self.conv4(out)
            out = self.classifier(out)
            return out
        except Exception as e:
            logging.error("Error occurred during forward propagation")
            raise CustomException(e, sys)
