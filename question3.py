import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np

from data_loader import loadDataH5
from utils import plot_accuracy

def resnet50(trainX, trainY, valX, valY):
    print(f'Single input shape: {valX[0].shape}')
    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(64, 64, 3))
    base_model.trainable = False  # Freeze base model
    print(f'ResNet50 Base model: {base_model.summary()}')

    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'ResNet50 Model: {model.summary()}')

    resnet50_model_result = model.fit(trainX, trainY,
                                              epochs=2,
                                              batch_size=32,
                                              validation_data=(valX, valY))

    print(f'ResNet50 history: {resnet50_model_result.history}')
    print(f'valX[0].shape: {valX[0].shape}')
    return model


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        trainX, trainY, valX, valY = loadDataH5()
        print(valX.shape)
        model = resnet50(trainX, trainY, valX, valY)
        single_input = np.expand_dims(valX[0], axis=0)
        print(f'single_input shape: {single_input.shape}')
        output = model.predict(single_input)
        print("Softmax output:", output)

