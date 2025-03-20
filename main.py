import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define paths containing the images
data_dir = "data/"
fake_dir = os.path.join(data_dir, "fake/")
real_dir = os.path.join(data_dir, "real/")

# Create directories for train, validation, and test sets
for subset in ['train', 'val', 'test']:
    for category in ['fake', 'real']:
        os.makedirs(os.path.join(data_dir, subset, category), exist_ok=True)


# Function to split data
def split_data(src_dir, train_dir, val_dir, test_dir, split_ratio):
    all_files = os.listdir(src_dir)
    np.random.shuffle(all_files)
    train_ratio = int(len(all_files) * split_ratio[0])
    val_ratio = int(len(all_files) * split_ratio[1])
    train_files = all_files[:train_ratio]
    val_files = all_files[train_ratio:train_ratio + val_ratio]
    test_files = all_files[train_ratio + val_ratio:]

    for file in train_files:
        shutil.copy(os.path.join(src_dir, file), train_dir)
    for file in val_files:
        shutil.copy(os.path.join(src_dir, file), val_dir)
    for file in test_files:
        shutil.copy(os.path.join(src_dir, file), test_dir)


# Split the data
split_ratio = [0.6, 0.2, 0.2]  # 60% train, 20% validation, 20% test
split_data(
    fake_dir,
    os.path.join(data_dir, 'train/fake'),
    os.path.join(data_dir, 'val/fake'),
    os.path.join(data_dir, 'test/fake'),
    split_ratio
)
split_data(
    real_dir,
    os.path.join(data_dir, 'train/real'),
    os.path.join(data_dir, 'val/real'),
    os.path.join(data_dir, 'test/real'),
    split_ratio
)

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

train_set = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_set = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_set = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define the CNN
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
