# Autoencoder Classifier

In this project, we create an autoencoder classifier for the CIFAR10 database with only half the data for the bird, deer and truck class. This classifier is composed of an encoder that extract features from the images and a CNN model that takes as input the incoded image and predict the image class.<br>   

## Getting Started

The jupyter notebook contains the development of the model. The `autoencoder_classifier.py` containes python script to train and test the model and the `extract_cifar10.py` is used to extract images from the CIFAR10 database.<br>
First you need to extract the [CIFAR10 database](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class.<br>
Before using this data we need to transform the data into images with shape (32, 32, 3) and save the images of each class in a separate folder. 

### Prerequisites

The jupyter notebook and the python scripts uses `python3`.
make sure to install all dependencies in `requirements.txt` file.

```bash
pip install --user --requirement requirements.txt
```

### Installing

To use the autoencoder classifier, make sure to do the following steps:

1.  Download [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
and extract it in the local directory
```bash
tar xvzf cifar-10-python.tar.gz -C /path/to/cloned/directory
```
2. Transform the batch files to images
```bash
python src/extract_cifar10.py cifar-10-batches-py/
```

### Use the model


1. To train the model use the `autoencoder_classifier.py` script<br>

for example to train the model `model.h5` for 10 epochs on train_data folder run the following script

```bash
python autoencoder_classifier.py -t -d train_data -m models/model.h5 --epochs 10
```
the new trained model will be saved at the current directory<br>

2. To predict classes of new images, first put some images in a folder and run
 ```bash
python autoencoder_classifier.py -p -d test/ -m models/model.h5
```
3. To evaluate the model on test data located at test_data/ folder run the command
 ```bash
python autoencoder_classifier.py -e -d test_data/ -m models/model.h5
```

for information about different options run
```bash
python autoencoder_classifier.py -h
```

## License

This project is licensed under the MIT License

