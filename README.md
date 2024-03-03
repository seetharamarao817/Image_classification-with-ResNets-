 **Image Classification with ResNets**

**Introduction**

This project demonstrates image classification using ResNet neural networks. We train a ResNet model on the CIFAR10 dataset and evaluate its performance on unseen data. We provide clear instructions on how to train, evaluate, and use the model.

**Installation**

To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

**Data**

We use the CIFAR10 dataset, which consists of 60,000 32x32 color images in 10 classes.

**Model Architecture**

We utilize ResNet-11, a variant of the ResNet architecture known for its depth and residual connections.

**Training Pipeline**

To train the model, run the following command:
```
python pipelines/training_pipeline.py
```

The training pipeline includes the following steps:

* Data loading and preprocessing
* Model initialization
* Training loop with data augmentation

The trained model is saved in the `artifacts` directory.

**Prediction Pipeline**

To predict labels for new images, run the following command:
```
python pipelines/prediction_pipeline.py
```

The prediction pipeline involves:

* Data loading and preprocessing
* Model loading
* Image classification

**User Interface**

We provide a user-friendly Streamlit application for model inference. To run it, run the following command:
```
streamlit run streamlit_app.py
```

The interface allows users to upload images and view the predicted labels.

**Screenshot of the User Interface**

![screenshot](https://github.com/seetharamarao817/Image_classification-with-ResNets-/blob/main/testimages/Screenshot%20(116).png)

**Usage**

**Training:**
1. Install requirements.
2. Run `python pipelines/training_pipeline.py`.

**Prediction:**
1. Install requirements.
2. Run `python pipelines/prediction_pipeline.py`.

**User Interface:**
1. Install requirements.
2. Run `streamlit run streamlit_app.py`. 
