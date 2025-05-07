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

    return trainX, trainY, valX, valY

if __name__ == "__main__":
    loadDataH5()


