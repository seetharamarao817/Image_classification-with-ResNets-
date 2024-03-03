import os
import tarfile
import sys
from src.logging.logging import logging
from src.exceptions.exceptions import CustomException

class CIFAR10DataLoader:
    def __init__(self, archive_path='./cifar10.tgz', extract_dir='./data'):
        self.archive_path = archive_path
        self.extract_dir = extract_dir

    def extract_data(self):
        try:
            logging.info("Extracting data from the archive")
            with tarfile.open(self.archive_path, 'r:gz') as tar:
                tar.extractall(path=self.extract_dir)
            logging.info("Extraction completed")
        except Exception as e:
            logging.error("Error occurred during data extraction")
            raise CustomException(e, sys)

    def get_classes(self):
        try:
            logging.info("Getting classes from the dataset")
            data_dir = os.path.join(self.extract_dir, 'cifar10')
            train_dir = os.path.join(data_dir, 'train')
            classes = os.listdir(train_dir)
            logging.info("Classes retrieved successfully")
            return classes
        except Exception as e:
            logging.error("Error occurred while getting classes")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        cifar_loader = CIFAR10DataLoader()
        cifar_loader.extract_data()
        classes = cifar_loader.get_classes()
        print("Classes:", classes)
    except CustomException as e:
        print("Custom Exception occurred:", e)
