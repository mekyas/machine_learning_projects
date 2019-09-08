# Autoencoder Classifier

In this project, we create an autoencoder classifier for the CIFAR10 database with only half the data for the bird, deer and truck class. This classifier is composed of an encoder that extract features from the images and a CNN model that takes as input the incoded image and predict the image class.<br>
The jupyter notebook contains all the steps we took to develop the model. 
Our first approach was to train the autoencoder separately before adding it to the CNN model, and train the classifer while freezing the autoencoder layers.
In The second approach, we train the autoencoder and the CNN model simultaneously. This model takes as input an image and outputs the decoded image and the image class. We used three different autoencoders architecture.<br>
The `autoencoder.py` containes python script to train and test the model and the `extract_cifar10.py` is used to extract images from the CIFAR10 database.   

## Getting Started

After cloning this repositery. First you need to extract the [CIFAR10 database](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class.<br>
Each of the batch files is a dictionary that contains a **data** element which is a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image classes are in the **labels** element which is a list of 10000 numbers in the range 0-9. <br>
Before using this data in our model we need to transform the data into images with shape (32, 32, 3) and save the images of each class in a separate folder. 

### Prerequisites

The jupyter notebook and the python scripts uses `python3`.
Run the follwing script in your terminal to install all dependencies.

```bash
pip install numpy scipy cython
pip install keras
pip install matplotlib
pip install scikit-image
pip install -U scikit-learn
pip install argparse
```

### Installing

To use the autoencoder classifier, make sure to do the following steps:

1. Clone the repository in your local machine or download the files separately

2.  Download [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
and extract it in the cloned directory
```bash
tar xvzf cifar-10-python.tar.gz -C /path/to/cloned/directory
```
3. Transform the batch files to images
```bash
python extract_cifar10.py cifar-10-batches-py/
```
4. To train the model use the `autoencoder.py` script<br>
for example to train the model `classifier_7_model.h5` for 10 epochs on train_data run the following script

```bash
python autoencoder.py -t -d train_data -m models/classifier/classifier_7_model.h5 --epochs 10
```
the new trained model will be saved at the current directory<br>

5. To predict classes of new images, first put some images in a folder and run the command
 ```bash
python autoencoder.py -p -d test/ -m models/classifier/classifier_7_model.h5
```
6. To evaluate the model on test data located at test_data/ folder run the command
 ```bash
python autoencoder.py -e -d test_data/ -m models/classifier/classifier_7_model.h5
```
for information about different options run
```bash
python autoencoder.py -h
```

## License

This project is licensed under the MIT License

