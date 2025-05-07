import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3

from data_loader import loadDataH5, preprocess_vgg
import numpy as np
import matplotlib.pyplot as plt

# Random Forest
def random_forest(trainX, trainY, valX, valY):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(trainX, trainY)
    y_pred_rf = rf.predict(valX)
    accuracy = accuracy_score(valY, y_pred_rf)
    print("Random Forest Accuracy:", accuracy)
    print(classification_report(valY, y_pred_rf))
    return accuracy

# Logistic Regression
def logistic_regression(trainX, trainY, valX, valY):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(trainX, trainY)
    y_pred_lr = lr.predict(valX)
    accuracy = accuracy_score(valY, y_pred_lr)
    print("Logistic Regression Accuracy:", accuracy)
    print(classification_report(valY, y_pred_lr))
    return accuracy

def preprocess_for_vgg(images):
    images = tf.image.resize(images, [64, 64])
    images = tf.cast(images, tf.float32)
    return tf.keras.applications.vgg16.preprocess_input(images)

# Preprocess all the images for ResNet50
def preprocess_for_resnet(images):
    resized = tf.image.resize(images, [64, 64])
    return tf.keras.applications.resnet50.preprocess_input(resized)

def preprocess_for_inception(images):
    resized = tf.image.resize(images, [75, 75])
    return tf.keras.applications.inception_v3.preprocess_input(resized)

def vgg16_as_feature_extractor(trainX, trainY, valX, valY):
    x_train_preprocessed = preprocess_for_vgg(trainX)
    x_val_preprocessed = preprocess_for_vgg(valX)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Extract features
    features_train = model.predict(x_train_preprocessed, batch_size=32, verbose=1)
    features_val = model.predict(x_val_preprocessed, batch_size=32, verbose=1)

    # Flatten features
    features_train_flat = features_train.reshape(features_train.shape[0], -1)
    features_val_flat = features_val.reshape(features_val.shape[0], -1)

    random_forest_ac = random_forest(features_train_flat, trainY, features_val_flat, valY)
    logistic_regression_ac = logistic_regression(features_train_flat, trainY, features_val_flat, valY)
    return random_forest_ac, logistic_regression_ac

def resnet50_as_feature_extractor(trainX, trainY, valX, valY):
    x_train_preprocessed = preprocess_for_resnet(trainX)
    x_val_preprocessed = preprocess_for_resnet(valX)

    # Load pretrained ResNet50 model (without top layer, output is 2048-dim vector)
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(64, 64, 3))

    # Extract features
    train_features = resnet.predict(x_train_preprocessed, batch_size=32, verbose=1)
    val_features = resnet.predict(x_val_preprocessed, batch_size=32, verbose=1)

    if len(trainY.shape) > 1 and trainY.shape[1] > 1:  # Flatten labels if one-hot encoded
        trainY = np.argmax(trainY, axis=1)
        valY = np.argmax(valY, axis=1)

    random_forest_ac = random_forest(train_features, trainY, val_features, valY)
    logistic_regression_ac = logistic_regression(train_features, trainY, val_features, valY)
    return random_forest_ac, logistic_regression_ac

def inceptionV3_as_feature_extractor(trainX, trainY, valX, valY):
    x_train_preprocessed = preprocess_for_inception(trainX)
    x_val_preprocessed = preprocess_for_inception(valX)

    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(75, 75, 3)
    )

    train_features = inception_model.predict(x_train_preprocessed, batch_size=32, verbose=1)
    val_features = inception_model.predict(x_val_preprocessed, batch_size=32, verbose=1)

    random_forest_ac = random_forest(train_features, trainY, val_features, valY)
    logistic_regression_ac = logistic_regression(train_features, trainY, val_features, valY)
    return random_forest_ac, logistic_regression_ac

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        class_names, trainX, trainY, valX, valY = loadDataH5()
        accuracies = {}

        vgg16_acc = {}
        vgg16_random_forest_ac, vgg16_logistic_regression_ac = vgg16_as_feature_extractor(trainX, trainY, valX, valY)
        vgg16_acc['random_forest'] = vgg16_random_forest_ac
        vgg16_acc['logistic_regression'] = vgg16_logistic_regression_ac

        resnet50_acc = {}
        resnet50_random_forest_ac, resnet50_logistic_regression_ac = resnet50_as_feature_extractor(trainX, trainY, valX,
                                                                                                   valY)
        resnet50_acc['random_forest'] = resnet50_random_forest_ac
        resnet50_acc['logistic_regression'] = resnet50_logistic_regression_ac

        inceptionV3_acc = {}
        inceptionV3_random_forest_ac, inceptionV3_logistic_regression_ac = inceptionV3_as_feature_extractor(trainX, trainY, valX,
                                                                                                   valY)
        inceptionV3_acc['random_forest'] = inceptionV3_random_forest_ac
        inceptionV3_acc['logistic_regression'] = inceptionV3_logistic_regression_ac

        accuracies['vgg16'] = vgg16_acc
        accuracies['resnet50'] = resnet50_acc
        accuracies['inceptionv3'] = inceptionV3_acc

        models = list(accuracies.keys())
        classifiers = ['random_forest', 'logistic_regression']

        # Extract accuracies in the correct order
        rf_scores = [accuracies[model]['random_forest'] for model in models]
        lr_scores = [accuracies[model]['logistic_regression'] for model in models]

        x = np.arange(len(models))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width / 2, rf_scores, width, label='Random Forest', color='cornflowerblue')
        bars2 = ax.bar(x + width / 2, lr_scores, width, label='Logistic Regression', color='salmon')

        # Add some text for labels, title and custom x-axis tick labels
        ax.set_ylabel('Accuracy')
        ax.set_title('CNN + Classifier Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([model.upper() for model in models])
        ax.set_ylim(0.7, 1.0)
        ax.legend()

        # Annotate each bar with accuracy value
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


