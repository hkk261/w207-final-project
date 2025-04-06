import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Define paths containing the images
data_dir = "data"
fake_dir = os.path.join(data_dir, "fake")
real_dir = os.path.join(data_dir, "real")

def get_dataset(fake_dir, real_dir, balance=False):
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]

    if balance:
        """Truncates each class to the minimum count"""
        min_images  = min(len(fake_images), len(real_images))
        fake_images = fake_images[:min_images]
        real_images = real_images[:min_images]
    
    image_paths = fake_images + real_images
    labels = ["fake"] * len(fake_images) + ["real"] * len(real_images)
    return image_paths, labels

def split_indices(indices, split_ratio):
    train = int(split_ratio[0] * len(indices))
    val = int(split_ratio[1] * len(indices))

    train_idx = indices[:train]
    val_idx = indices[train:train + val]
    test_idx = indices[train + val:]
    return train_idx, val_idx, test_idx

def split_data(image_paths, labels, split_ratio, seed=42, balance=False):
    image_paths, labels = np.array(image_paths), np.array(labels)
    
    np.random.seed(seed)
    
    if not balance:
        idx = np.arange(len(image_paths))
        np.random.shuffle(idx)
        train_idx, val_idx, test_idx = split_indices(idx, split_ratio)
        
    else:
        """Balance class by class"""
        fake_idx = np.where(labels == "fake")[0]
        real_idx = np.where(labels == "real")[0]
        
        np.random.shuffle(fake_idx)
        np.random.shuffle(real_idx)

        fake_train, fake_val, fake_test = split_indices(fake_idx, split_ratio)
        real_train, real_val, real_test = split_indices(real_idx, split_ratio)
        
        train_idx = np.concatenate([fake_train, real_train])
        val_idx = np.concatenate([fake_val, real_val])
        test_idx = np.concatenate([fake_test, real_test])
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

    return (image_paths[train_idx], labels[train_idx]), (image_paths[val_idx], labels[val_idx]), (image_paths[test_idx], labels[test_idx])

def print_dataset(name, labels_array):
    print(f"{name} Dataset: {len(labels_array)}")
    print(f"fake: {np.sum(labels_array == 'fake')}")
    print(f"real: {np.sum(labels_array == 'real')}")

image_paths, labels = get_dataset(fake_dir, real_dir, balance=True)
(train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(image_paths, labels, split_ratio=[0.6, 0.2, 0.2], balance=True)

print_dataset("Train", train_y)
print_dataset("Validation", val_y)
print_dataset("Test", test_y)
print_dataset("Full", np.concatenate([train_y, val_y, test_y]))

train_df = pd.DataFrame({"image_path": train_x, "label": train_y})
val_df   = pd.DataFrame({"image_path": val_x,   "label": val_y})
test_df  = pd.DataFrame({"image_path": test_x,  "label": test_y})

# Image properties
img_height, img_width = 30, 30
batch_size = 32

# # Data augmentation and loading
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# val_test_datagen = ImageDataGenerator(rescale=1./255)

# train_set = train_datagen.flow_from_dataframe(
#     dataframe=train_df,
#     x_col="image_path",
#     y_col="label",
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary'
# )

# val_set = val_test_datagen.flow_from_dataframe(
#     dataframe=val_df,
#     x_col="image_path",
#     y_col="label",
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary'
# )

# test_set = val_test_datagen.flow_from_dataframe(
#     dataframe=test_df,
#     x_col="image_path",
#     y_col="label",
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False 
# )

# # Define the CNN
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     MaxPooling2D(pool_size=2, strides=2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=2, strides=2),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=2, strides=2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Binary classification
# ])

# # Compile the model
# learning_rate = 0.0001
# model.compile(
#     optimizer=Adam(learning_rate=learning_rate),
#     loss='binary_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )

# # Custom early stopping callback
# class CustomEarlyStopping(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         stop_acc = 0.9
#         train_acc = logs.get('accuracy')
#         val_acc = logs.get('val_accuracy')
#         if train_acc >= stop_acc and val_acc >= stop_acc:
#             print(f"\nStopping early at epoch {epoch + 1} - "
#                   f"train accuracy: {train_acc:.4f}, validation accuracy: {val_acc:.4f}")
#             self.model.stop_training = True

# # Train the model
# max_epochs = 50
# history = model.fit(
#     train_set,
#     validation_data=val_set,
#     epochs=max_epochs,
#     callbacks=[CustomEarlyStopping()]
# )

# test_loss, test_acc, test_auc = model.evaluate(test_set)
# print(f"Test Accuracy: {test_acc:.4f}")
# print(f"Test AUC: {test_auc:.4f}")

# test_probs = model.predict(test_set, verbose=0) 
# test_preds = (test_probs.ravel() >= 0.5).astype(int)  
# y_true = test_set.classes  

# f1 = f1_score(y_true, test_preds)
# print(f"F1 score: {f1:.4f}")

# # Plotting ROC curve
# fpr, tpr, thresholds = roc_curve(y_true, test_probs.ravel())
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f"AUROC = {test_auc:.4f}")
# plt.plot([0, 1], [0, 1], 'k--') 
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("AUROC")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Plotting confusion matrix
# cm = confusion_matrix(y_true, test_preds)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()


def flatten_image(path, size=(img_height, img_width)):
    img = load_img(path, target_size=size)
    array = img_to_array(img) / 255.0
    return array.flatten()

def prepare_lr_data(image_paths, labels, size=(img_height, img_width)):
    X, Y = [], []
    for path in image_paths:
        image = flatten_image(path, size)
        X.append(image)

    for label in labels:
        if label == "fake":
            Y.append(0)
        else:
            Y.append(1)
    
    return np.array(X), np.array(Y)

x_train, y_train = prepare_lr_data(train_x, train_y)
x_val, y_val = prepare_lr_data(val_x, val_y)
x_test, y_test = prepare_lr_data(test_x, test_y)



# lr_model = LogisticRegression(max_iter=250, solver='lbfgs')
# lr_model.fit(x_train, y_train)

# y_val_pred = lr_model.predict(x_val)
# val_acc = accuracy_score(y_val, y_val_pred)
# val_auc = roc_auc_score(y_val, lr_model.predict_proba(x_val)[:, 1])
# val_f1 = f1_score(y_val, y_val_pred)

# print("Validation Performance:")
# print(f"Accuracy: {val_acc:.4f}")
# print(f"AUC: {val_auc:.4f}")
# print(f"F1 Score: {val_f1:.4f}")

# y_test_pred = lr_model.predict(x_test)
# test_acc = accuracy_score(y_test, y_test_pred)
# test_auc = roc_auc_score(y_test, lr_model.predict_proba(x_test)[:, 1])
# test_f1 = f1_score(y_test, y_test_pred)

# print("Test Performance:")
# print(f"Accuracy: {test_acc:.4f}")
# print(f"AUC: {test_auc:.4f}")
# print(f"F1 Score: {test_f1:.4f}")




def build_model(num_features, learning_rate):
    """Build a TF binary logistic regression model using Keras.
    
    Args:
      num_features: Number of input features.
      learning_rate: Desired learning rate for SGD.
    
    Returns:
      model: A compiled tf.keras model.
    """
    # Clear any previous TF graphs and set a seed for reproducibility.
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    
    # Build a Sequential model with a single Dense layer.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        units=1,                          # output dimension is 1 (binary classification)
        input_shape=(num_features,),      # input dimension
        use_bias=True,
        activation='sigmoid',             # sigmoid for binary logistic regression
        kernel_initializer=tf.keras.initializers.Ones(),  # initialize weights to 1
        bias_initializer=tf.keras.initializers.Ones()     # initialize bias to 1
    ))
    
    # Use SGD as the optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Compile the model using binary crossentropy loss and accuracy as a metric.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


tf.random.set_seed(0)
# 2. Build and compile model
# YOUR CODE HERE
learning_rate = 0.001
num_features = x_train.shape[1]
model_tf = build_model(num_features, learning_rate)

# 3. Fit the model
# YOUR CODE HERE
num_epochs = 1000
batch_size = 32
history = model_tf.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)

y_probs_tf = model_tf.predict(x_test).ravel()
y_preds_tf = (y_probs_tf >= 0.5).astype(int)

acc_tf = accuracy_score(y_test, y_preds_tf)
auc_tf = roc_auc_score(y_test, y_probs_tf)
f1_tf = f1_score(y_test, y_preds_tf)

print("Logistic Regression performance:")
print(f"Test Accuracy: {acc_tf:.4f}")
print(f"Test AUC: {auc_tf:.4f}")
print(f"F1 Score: {f1_tf:.4f}")