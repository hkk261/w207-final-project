import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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
img_height, img_width = 300, 300
batch_size = 32

# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_set = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_set = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False 
)

# Define the CNN
'''
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=1, activation='sigmoid'))  # Binary classification
'''


# Baseline FFNN
model = Sequential()

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


# Compile the model
learning_rate = 0.0001
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Early stopping callback: stops training when both train and validation accuracy are high enough
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        stop_acc = 0.9
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc >= stop_acc and val_acc >= stop_acc:
            print(f"\nStopping early at epoch {epoch + 1} - train accuracy: {train_acc:.4f}, validation accuracy: {val_acc:.4f}")
            self.model.stop_training = True


# Train the model
max_epochs = 50
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=max_epochs,
    callbacks=[CustomEarlyStopping()]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_set)
print(f"Test Accuracy: {test_acc:.4f}")