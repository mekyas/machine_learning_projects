import pickle
import glob
import os
import sys
import numpy as np
from skimage.io import imsave


data_dir = "train_data"
test_dir = "test_data"
train_label_file = "train_labels.txt"
test_label_file = "test_labels.txt"


def unpack_file(fname):

    with open(fname, "rb") as f:
        result = pickle.load(f, encoding="bytes")

    return result

def save_as_image(img_flat, categorie, fname, dirname):

    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))

    imsave(os.path.join(dirname, categorie, fname), img)


def main(file_dir):

    train_labels = {}
    test_labels = {}
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for c in classes:
        os.makedirs(os.path.join(data_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)
        
    for fname in glob.glob(os.path.join(file_dir, "*_batch*")):
        data = unpack_file(fname)
        print("start unpacking file {}".format(fname))
        for i in range(10000):
            img_flat = data[b"data"][i]
            imgname = data[b"filenames"][i].decode("utf-8")
            label = data[b"labels"][i]
            categorie = classes[label]
            if fname == os.path.join(file_dir, "test_batch"):
                save_as_image(img_flat, categorie, imgname, test_dir)
                test_labels[imgname] = label
            else:
                save_as_image(img_flat, categorie, imgname, data_dir)
                train_labels[imgname] = label

    with open(train_label_file, "w") as f:
        for (fname, label) in train_labels.items():
            f.write("{0} {1}\n".format(fname, label))
    
    with open(test_label_file, "w") as f:
        for (fname, label) in test_labels.items():
            f.write("{0} {1}\n".format(fname, label))


if __name__ == "__main__":
    file_dir = sys.argv[1]
    main(file_dir)