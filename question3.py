import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

from data_loader import loadDataH5

MODEL_PATH = "models"

def resnet50_model(epochs_count, trainX, trainY, valX, valY):
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



    phaseA_result = model.fit(trainX,
                                      trainY,
                                      epochs=epochs_count,
                                      batch_size=32,
                                      validation_data=(valX, valY))
    print(f'ResNet50 phase A history: {phaseA_result.history}')

    base_model.trainable = True
    trainable_flag = False

    for layer in base_model.layers:
        if layer.name == 'conv5_block3_1_conv':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'ResNet50 Model: {model.summary()}')

    checkpoint = ModelCheckpoint(
        filepath=f"{MODEL_PATH}/resnet50.h5",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

    resnet50_model_result = model.fit(trainX,
                                      trainY,
                                      epochs=epochs_count,
                                      batch_size=32,
                                      validation_data=(valX, valY),
                                      callbacks=[checkpoint])

    print(f'ResNet50 history: {resnet50_model_result.history}')

    resnet50_preds = np.argmax(model.predict(valX), axis=1)
    resnet50_acc = accuracy_score(valY, resnet50_preds)
    print(f'resnet50_acc: {resnet50_acc}')

    return resnet50_acc

def vgg16_model(epochs_count, trainX, trainY, valX, valY):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False  # Freeze base model
    print(f'VGG16 Base model: {base_model.summary()}')

    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'VGG16 Model: {model.summary()}')

    phaseA_result = model.fit(trainX,
                              trainY,
                              epochs=epochs_count,
                              batch_size=32,
                              validation_data=(valX, valY))
    print(f'VGG16 phase A history: {phaseA_result.history}')

    base_model.trainable = True
    trainable_flag = False

    for layer in base_model.layers:
        if layer.name == 'block4_conv1':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        filepath=f"{MODEL_PATH}/vgg16.h5",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

    vgg16_model_result = model.fit(trainX,
                              trainY,
                              epochs=epochs_count,
                              batch_size=32,
                              validation_data=(valX, valY),
                              callbacks=[checkpoint])

    print(f'VGG16 history: {vgg16_model_result.history}')

    vgg16_preds = np.argmax(model.predict(valX), axis=1)
    vgg16_acc = accuracy_score(valY, vgg16_preds)
    print(f'vgg16_acc: {vgg16_acc}')

    return vgg16_acc

def soft_voting_ensemble(valX, valY):
    vgg16 = load_model(f"{MODEL_PATH}/vgg16.h5")
    vgg16_prediction = vgg16.predict(valX)
    print(f'vgg16 prediction values: {vgg16_prediction}')
    print(f'vgg16 prediction type: {type(vgg16_prediction)}')

    resnet50 = load_model(f"{MODEL_PATH}/resnet50.h5")
    resnet50_prediction = resnet50.predict(valX)
    print(f'resnet50 prediction values: {resnet50_prediction}')
    print(f'resnet50 prediction type: {type(resnet50_prediction)}')

    ensemble_preds = (vgg16_prediction + resnet50_prediction) / 2
    print(f'final prediction values: {ensemble_preds}')

    ensemble_acc = accuracy_score(valY, ensemble_preds)
    print(f'ensemble_acc: {ensemble_acc}')

    return ensemble_acc

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        trainX, trainY, valX, valY = loadDataH5()
        resnet50_acc = resnet50_model(2, trainX, trainY, valX, valY)
        vgg16_acc = vgg16_model(2, trainX, trainY, valX, valY)
        soft_voting_ensemble_acc = soft_voting_ensemble(valX, valY)

        print(f"VGG16 Accuracy:     {vgg16_acc:.4f}")
        print(f"ResNet50 Accuracy:  {resnet50_acc:.4f}")
        print(f"Ensemble(Soft Voting) Accuracy:  {soft_voting_ensemble_acc:.4f}")




