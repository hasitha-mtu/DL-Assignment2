import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from data_loader import loadDataH5
from utils import plot_accuracy

# Choose one of the base models
def get_base_model(name="VGG16", input_shape=(224, 224, 3)):
    if name == "VGG16":
        return VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "ResNet50":
        return ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "InceptionV3":
        input_shape = (299, 299, 3)  # InceptionV3 expects 299x299
        return InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Model name not recognized.")

# Build the complete model
def build_transfer_model(base_model_name="VGG16", num_classes=10):
    input_shape = (75, 75, 3) if base_model_name == "InceptionV3" else (64, 64, 3)
    base_model = get_base_model(base_model_name, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model
    print(f'Base model: {base_model.summary()}')

    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Preprocessing specific to each model
    if base_model_name == "VGG16":
        x = tf.keras.applications.vgg16.preprocess_input(x)
    elif base_model_name == "ResNet50":
        x = tf.keras.applications.resnet.preprocess_input(x)
    elif base_model_name == "InceptionV3":
        x = tf.keras.applications.inception_v3.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(f'Model: {model.summary()}')
    return model

def vgg16_phase_A(epoch_count, trainX, trainY, valX, valY):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False  # Freeze base model
    print(f'Base model: {base_model.summary()}')

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

    print(f'Model: {model.summary()}')

    model_result = model.fit(trainX, trainY,
                             epochs=epoch_count,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(model_result)
    return model_result

def vgg16_phase_B(epoch_count, trainX, trainY, valX, valY):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False  # Freeze base model
    print(f'Base model: {base_model.summary()}')

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

    print(f'Model: {model.summary()}')

    #  For Phase B we need to go through Phase A first
    _model_result = model.fit(trainX, trainY,
                             epochs=epoch_count,
                             batch_size=32,
                             validation_data=(valX, valY))

    base_model.trainable = True
    trainable_flag = False

    # Unfreeze last conv layers of the VGG16

    for layer in base_model.layers:
        if layer.name == 'block4_conv1':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'Model: {model.summary()}')

    model_result = model.fit(trainX, trainY,
                             epochs=epoch_count,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(model_result)
    return model_result

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        _, trainX, trainY, valX, valY = loadDataH5()
        vgg16_phase_A(25, trainX, trainY, valX, valY)
