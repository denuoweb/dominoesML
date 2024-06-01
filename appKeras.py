
import tensorflow as tf
import os
import re
from tensorflow.keras import layers, models

base_dir = 'basePhotos'
train_dir = os.path.join(base_dir, 'train')
validate_dir = os.path.join(base_dir, 'validate')
test_dir = os.path.join(base_dir, 'test')

def parse_function(filename):
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [150, 150])
    # Normalize the image
    image_normalized = image_resized / 255.0
    return image_normalized

def extract_label(file_path):
    # Extracts the label from the file path (assuming label is part of the filename before the first underscore)
    base = os.path.basename(file_path)
    label = int(re.match(r"(\d+)", base).group(1))
    return label

def get_dataset(data_dir):
    file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    labels = [extract_label(fp) for fp in file_paths]
    # Convert lists to tensors
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels, dtype=tf.float32)
    # Create a dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    # Map the parse_function over the dataset
    dataset = dataset.map(lambda filename, label: (parse_function(filename), label))
    return dataset

# Use the new get_dataset function to create datasets
train_dataset = get_dataset(os.path.join(base_dir, 'train')).batch(10)
validate_dataset = get_dataset(os.path.join(base_dir, 'validate')).batch(2)
test_dataset = get_dataset(os.path.join(base_dir, 'test')).batch(1)


# Data augmentation is done on the fly during model training
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.Rescaling(1./255),
], name='data_augmentation')

model = models.Sequential([
    # Data augmentation layers
    #layers.experimental.preprocessing.RandomFlip("horizontal"),
    #layers.experimental.preprocessing.RandomRotation(0.1),
    # Rescaling layer
    #layers.experimental.preprocessing.Rescaling(1./255),
    # Convolutional and pooling layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(.25),
    # Flatten and dense layers for regression
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # No activation for regression output
])

optimizer = tf.keras.optimizers.Adam(learning_rate=.004)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Callbacks are used to guide the training process
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500),
    #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25)
]

model.fit(
    train_dataset,
    epochs=5000,
    validation_data=validate_dataset,
    callbacks=callbacks
)

test_loss, test_mae = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

model.save('domino_dot_counter.h5')