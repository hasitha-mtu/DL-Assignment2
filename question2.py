import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from data_loader import loadDataH5
from utils import plot_accuracy

def vgg16_phase_A(trainX, trainY, valX, valY):
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

    vgg16_phase_A_model_result = model.fit(trainX, trainY,
                             epochs=25,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(vgg16_phase_A_model_result, "vgg16_phase_A_model_result",True)

def vgg16_phase_B_1(trainX, trainY, valX, valY):
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

    print(f'VGG16 Model: {model.summary()}')

    #  For Phase B we need to go through Phase A first
    _model_result = model.fit(trainX, trainY,
                             epochs=25,
                             batch_size=32,
                             validation_data=(valX, valY))

    base_model.trainable = True
    trainable_flag = False

    # Unfreeze last conv layers of the VGG16

    for layer in base_model.layers:
        if layer.name == 'block5_conv1':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'VGG16 Model(block5_conv1): {model.summary()}')

    vgg16_phase_B_1_model_result = model.fit(trainX, trainY,
                             epochs=50,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(vgg16_phase_B_1_model_result, "vgg16_phase_B_1_model_result", True)

def vgg16_phase_B_2(trainX, trainY, valX, valY):
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

    print(f'Model: {model.summary()}')

    #  For Phase B we need to go through Phase A first
    _model_result = model.fit(trainX, trainY,
                             epochs=25,
                             batch_size=32,
                             validation_data=(valX, valY))

    base_model.trainable = True
    trainable_flag = False

    # Unfreeze block5_conv1 conv layers of the VGG16
    for layer in base_model.layers:
        if layer.name == 'block5_conv1':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'VGG16 Model(unfreeze block5_conv1): {model.summary()}')

    _ = model.fit(trainX, trainY,
                             epochs=50,
                             batch_size=32,
                             validation_data=(valX, valY))

    # Unfreeze block4_conv1 conv layers of the VGG16
    for layer in base_model.layers:
        if layer.name == 'block4_conv1':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'VGG16 Model(unfreeze block4_conv1): {model.summary()}')

    vgg16_phase_B_2_model_result = model.fit(trainX, trainY,
                             epochs=75,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(vgg16_phase_B_2_model_result, "vgg16_phase_B_2_model_result", True)

def resnet50_phase_A(trainX, trainY, valX, valY):
    base_model = ResNet50(weights="imagenet", include_top=False,  pooling='avg', input_shape=(64, 64, 3))
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

    resnet50_phase_A_model_result = model.fit(trainX, trainY,
                             epochs=25,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(resnet50_phase_A_model_result, "resnet50_phase_A_model_result", True)

def resnet50_phase_B_1(trainX, trainY, valX, valY):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(64, 64, 3))
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
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
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

    print(f'Model: {model.summary()}')

    #  For Phase B we need to go through Phase A first
    _model_result = model.fit(trainX, trainY,
                             epochs=25,
                             batch_size=32,
                             validation_data=(valX, valY))

    base_model.trainable = True
    trainable_flag = False

    # Unfreeze last conv layers of the ResNet50

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

    resnet50_phase_B_1_model_result = model.fit(trainX, trainY,
                             epochs=50,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(resnet50_phase_B_1_model_result, "resnet50_phase_B_1_model_result", True)

def resnet50_phase_B_2(trainX, trainY, valX, valY):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
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

    #  For Phase B we need to go through Phase A first
    _model_result = model.fit(trainX, trainY,
                             epochs=25,
                             batch_size=32,
                             validation_data=(valX, valY))

    base_model.trainable = True
    trainable_flag = False

    # Unfreeze conv5_block3_1_conv conv layers of the VGG16
    for layer in base_model.layers:
        if layer.name == 'conv5_block3_1_conv':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'ResNet50 Model(unfreeze conv5_block3_1_conv): {model.summary()}')

    _ = model.fit(trainX, trainY,
                             epochs=50,
                             batch_size=32,
                             validation_data=(valX, valY))

    # Unfreeze conv5_block1_1_conv conv layers of the ResNet50
    for layer in base_model.layers:
        if layer.name == 'conv5_block1_1_conv':
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'ResNet50 Model(unfreeze conv5_block1_1_conv): {model.summary()}')

    resnet50_phase_B_2_model_result = model.fit(trainX, trainY,
                             epochs=75,
                             batch_size=32,
                             validation_data=(valX, valY))

    plot_accuracy(resnet50_phase_B_2_model_result, "resnet50_phase_B_2_model_result", True)

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        trainX, trainY, valX, valY = loadDataH5()
        print("=======================================================================================================")
        vgg16_phase_A(trainX, trainY, valX, valY)
        print("=======================================================================================================")
        vgg16_phase_B_1(trainX, trainY, valX, valY)
        print("=======================================================================================================")
        vgg16_phase_B_2(trainX, trainY, valX, valY)
        print("=======================================================================================================")
        resnet50_phase_A(trainX, trainY, valX, valY)
        print("=======================================================================================================")
        resnet50_phase_B_1(trainX, trainY, valX, valY)
        print("=======================================================================================================")
        resnet50_phase_B_2(trainX, trainY, valX, valY)
        print("=======================================================================================================")
