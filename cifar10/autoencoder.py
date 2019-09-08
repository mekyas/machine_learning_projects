import argparse
import warnings
import os

AUTO_CNN = "models/auto_cnn.h5"
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 25
DECAY = 1e-2
SPE = 500
nb_test = 10000
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def build_parser():
    """parse command line argument"""

    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--model', type=str,
                            dest='model_dir', required=True, 
                            help='path to model file', default=AUTO_CNN)
    
    parser.add_argument('-t', '--train', action='store_true',
                        dest='is_train', help='Activate training flag',
                        default=False)
    
    parser.add_argument('-p', '--predict', action='store_true',
                        dest='is_test', help='Activate predict flag',
                        default=False)
    
    parser.add_argument('-e', '--evaluate', action='store_true',
                        dest='is_eval', help='Activate evaluation flag',
                        default=False)

    parser.add_argument('-d','--data', type=str,
                        dest='data', help='path to images folder (it can be train data or test data)',
                        required=True)

    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint', help='path to save the trained model',
                        default="trained_model.h5")

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        default=BATCH_SIZE)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        default=LEARNING_RATE)
    
    parser.add_argument('--steps-per-epoch', type=float,
                        dest='steps_per_epoch',
                        help='steps per epoch (default %(default)s)',
                        default=SPE)

    parser.add_argument('--version', action='version',
	                    version='%(prog)s 1.0')

    return parser



def fit_model(params):
    from keras import models
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint
    
    cnn = models.load_model(params.model_dir)

    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    train_img_generator = train_datagen.flow_from_directory(
        params.data,
        target_size=(32, 32),
        batch_size=params.batch_size)

    def data_generator(image_gen=train_img_generator):
        for X, y in image_gen:
            yield (X, {"autoencoder": X, "classifier_out": y})

    model_check = ModelCheckpoint(params.checkpoint, 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=True, 
                                save_weights_only=False)
    
    classifier_history = cnn.fit_generator(data_generator(),
                            steps_per_epoch = params.steps_per_epoch,                        
                            epochs=params.epochs,
                            callbacks=[model_check])

def predict(params):
    from keras import models
    import imghdr
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np

    cnn = models.load_model(params.model_dir)

    test_data = []
    for file in os.listdir(params.data):
        f = os.path.join(params.data, file)
        if imghdr.what(f) is not None:
            img = imread(f)
            if img.shape != (32, 32, 3):
                img = resize(img, (32, 32, 3))
            test_data.append(img)
    test = np.stack(test_data, axis=0)
    test = test/255.
    
    _, predict = cnn.predict(test,batch_size=1)
    predict = np.argmax(predict, axis=1)
    i = 0
    with open("images_classes.txt", 'w') as f:
        f.write("File name \t Class\n")
        for file in os.listdir(params.data):
            img = os.path.join(params.data, file)
            if imghdr.what(img) is not None:
                print("the file {} is an image of {}".format(file, classes[predict[i]]))
                f.write("{} \t {}\n".format(file, classes[predict[i]]))
                i +=1
    
def evaluate(params):
    from keras import models
    from keras.preprocessing.image import ImageDataGenerator

    cnn = models.load_model(params.model_dir)

    test_datagen = ImageDataGenerator(
        rescale=1./255)

    test_img_generator = test_datagen.flow_from_directory(
        params.data,
        target_size=(32, 32),
        batch_size=1, 
        shuffle=False)

    def data_generator(image_gen=test_img_generator):
        for X, y in image_gen:
            yield (X, {"autoencoder": X, "classifier_out": y})
    
    
    filenames = test_img_generator.filenames
    nb_samples = len(filenames)

    result = cnn.evaluate_generator(data_generator(),steps =nb_samples)
    print("The accuracy of the test set is {0:.2f}%".format(result[3]*100))

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = build_parser()
    params = parser.parse_args()
    if params.is_train:
        fit_model(params)
    elif params.is_test:
        predict(params)
    elif params.is_eval:
        evaluate(params)
    else:
        print("Please choose whither to perform training, prediction or evaluation")