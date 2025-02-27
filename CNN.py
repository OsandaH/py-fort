import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('bird.jpg')
dimensions = img.shape
print(dimensions) #Height #width #No of chanels

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
red, green, blue = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]

plt.imshow(red, cmap='Reds')
plt.title("Red")
plt.show()

plt.imshow(green, cmap='Greens')
plt.title("Green")
plt.show()

plt.imshow(blue, cmap='Blues')
plt.title("Blue")
plt.show()

#Grayscale
Gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to RGB format
plt.imshow(Gray_img, cmap = 'gray')
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the CNN architecture
model = models.Sequential()

# Convolutional and pooling layers
model.add(layers.ZeroPadding2D(padding=(2, 2), input_shape=(28, 28, 1))) 
model.add(layers.Conv2D(32, (3, 3), strides=(1, 2), activation='relu'))
model.add(layers.AveragePooling2D((3, 3)))

model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), strides=(1, 2), activation='relu'))
model.add(layers.AveragePooling2D((3, 3)))

model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), strides=(1, 2), activation='relu'))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=0, validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_acc * 100:.2f}%')



#### CNN Hyperparameter Tuning ####

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt  # Hyperparameter tuning
import numpy as np

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define the hyperparameter tuning function
def build_model(hp):
    model = models.Sequential()

    # Hyperparameter tuning for number of filters
    filters_1 = hp.Choice('filters_1', [32, 64, 128, 256])
    filters_2 = hp.Choice('filters_2', [32, 64, 128, 256])
    
    # Convolutional and pooling layers
    model.add(layers.Conv2D(filters_1, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(filters_2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters_2, (3, 3), activation='relu'))

    # Dropout to prevent overfitting
    dropout_rate = hp.Choice('dropout_rate', [0.2, 0.3, 0.5])
    model.add(layers.Dropout(dropout_rate))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10 classes for CIFAR-10

    # Hyperparameter tuning for learning rate
    learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

# Use Keras Tuner to find the best hyperparameters
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=5,
                     factor=3,
                     directory='tuner_results',
                     project_name='cifar10_tuning')

# Perform hyperparameter search
tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), batch_size=64)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best Hyperparameters:\n Filters 1: {best_hps.get('filters_1')}, "
      f"Filters 2: {best_hps.get('filters_2')}, "
      f"Dropout Rate: {best_hps.get('dropout_rate')}, "
      f"Learning Rate: {best_hps.get('learning_rate')}")

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(train_images, train_labels, epochs=15, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate on test data
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=1 )
print(f'Best Model Test Accuracy: {test_acc * 100:.2f}%')

