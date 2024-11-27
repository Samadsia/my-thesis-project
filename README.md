import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # First Convolutional Layer
    layers.MaxPooling2D((2, 2)),                                           # Pooling Layer
    layers.Conv2D(64, (3, 3), activation='relu'),                          # Second Convolutional Layer
    layers.MaxPooling2D((2, 2)),                                           # Pooling Layer
    layers.Conv2D(64, (3, 3), activation='relu'),                          # Third Convolutional Layer
    layers.Flatten(),                                                      # Flattening Layer
    layers.Dense(64, activation='relu'),                                   # Fully Connected Layer
    layers.Dense(10, activation='softmax')                                 # Output Layer (10 classes, I can adjust if needed)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Prepare Dummy Data (or I can load my dataset)
train_data = np.random.random((100, 32, 32, 3))   # 100 samples of 32x32 RGB images
train_labels = np.random.randint(10, size=(100,)) # 100 labels (0-9 for 10 classes)
test_data = np.random.random((20, 32, 32, 3))     # 20 test samples
test_labels = np.random.randint(10, size=(20,))   # 20 test labels

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)

# Plot training and validation loss
plt.figure(figsize=(12, 6))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
