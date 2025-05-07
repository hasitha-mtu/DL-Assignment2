import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import xgboost as xgb

from data_loader import loadDataH5, preprocess_vgg

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        class_names, trainX, trainY, valX, valY = loadDataH5()
        x_train_preprocessed = preprocess_vgg(trainX)
        x_val_preprocessed = preprocess_vgg(valX)

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        model = Model(inputs=base_model.input, outputs=base_model.output)

        # Extract features
        features_train = model.predict(x_train_preprocessed, batch_size=32, verbose=1)
        features_val = model.predict(x_val_preprocessed, batch_size=32, verbose=1)

        # Flatten features
        features_train_flat = features_train.reshape(features_train.shape[0], -1)
        features_val_flat = features_val.reshape(features_val.shape[0], -1)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(features_train_flat, trainY)
        y_pred_lr = lr.predict(features_val_flat)
        print("Logistic Regression Accuracy:", accuracy_score(valY, y_pred_lr))
        print(classification_report(valY, y_pred_lr))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(features_train_flat, trainY)
        y_pred_rf = rf.predict(features_val_flat)
        print("Random Forest Accuracy:", accuracy_score(valY, y_pred_rf))
        print(classification_report(valY, y_pred_rf))

        # SVC
        svm_clf = SVC(kernel='rbf', C=10, gamma='scale')  # Tune these if needed
        svm_clf.fit(trainX, trainY)
        y_pred_svm = svm_clf.predict(valX)
        print("SVM Accuracy:", accuracy_score(valY, y_pred_svm))
        print(classification_report(valY, y_pred_svm))

        # XGB
        xgb_clf = xgb.XGBClassifier(tree_method='hist', n_estimators=100, use_label_encoder=False,
                                    eval_metric='mlogloss')
        xgb_clf.fit(trainX, trainY)
        y_pred_xgb = xgb_clf.predict(valX)
        print("XGBoost Accuracy:", accuracy_score(valY, y_pred_xgb))
        print(classification_report(valY, y_pred_xgb))

        # Optional: Classification Report
        print("\nLogistic Regression Report:\n", classification_report(valY, y_pred_lr, target_names=class_names))

        cm = confusion_matrix(valY, y_pred_rf)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title("Random Forest Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

