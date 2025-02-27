import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

data = load_diabetes()
X,y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

## Build DNN model ##

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),    
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.3),    
    tf.keras.layers.Dense(1)
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(20,activation='tanh',input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(10,activation='tanh'),
#     tf.keras.layers.Dense(8,activation='tanh'),
#     tf.keras.layers.Dense(4,activation='tanh'),
#     tf.keras.layers.Dense(1)
# ])

## activation fn - tanh, softmax, relu

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense,Input
# model = Sequential([
#     Input(shape = (X_train.shape[1],)),
#     Dense(32, activation = 'relu'),
#     Dense(16, activation = 'relu'),
#     Dense(1)
# ])

# optimizer 
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['mae'])
#model.compile(optimizer = optimizer, loss = 'mean_squared_logarithmic_error', metrics = ['mae'])

### for classifiers 
#model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['mae'])
#model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['mae'])
#model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['mae'])


model.fit(X_train, y_train, epochs = 10, batch_size = 8, validation_data = (X_test, y_test))

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Mean Absolute Error: {test_mae:.4f}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'g')
plt.show()


#### Hyperparameter Tuning ####

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Choice('num_neurons', [32, 64, 128]),
                                    activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh']),
                                    input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_rate', [0.2, 0.3, 0.5])))
    model.add(tf.keras.layers.Dense(hp.Choice('num_neurons', [32, 64, 128]),
                                    activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])))
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_rate', [0.2, 0.3, 0.5])))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad']),
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# Set up tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=10,
    directory='tuning_results',
    project_name='diabetes_regression'
)

# Run hyperparameter search
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=1)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=best_hps.get('epochs', 50), batch_size=best_hps.get('batch_size', 32), validation_data=(X_test, y_test), verbose=1)

test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Mean Absolute Error: {test_mae:.4f}')

# Make predictions
y_pred = best_model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g')
plt.show()



### Heat Map ###

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(y_test, y_pred, target_names=data.target_names)
print("Classification Report:")
print(class_report)
print(conf_matrix)
