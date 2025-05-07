import numpy as np
import h5py
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

DATA_PATH = "datasets/earth_data.h5"

def loadDataH5():
    with h5py.File(DATA_PATH,'r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        print (valX.shape,valY.shape)

        labels = np.array(hf['trainY'])

    return labels, trainX, trainY, valX, valY

# Convert to float32 and resize to 224x224
def preprocess_vgg(images):
    images = tf.image.resize(images, [64, 64])  # Resize to match VGG16 input
    images = tf.cast(images, tf.float32)  # Ensure float32 type
    return preprocess_input(images)  # Normalize for VGG16 (subtract mean RGB, scale)

if __name__ == "__main__":
    trainX, trainY, valX, valY = loadDataH5()
    test = preprocess_vgg(trainX)


