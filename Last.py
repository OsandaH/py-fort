import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras_tuner as kt


# Load the Boston Housing dataset
boston = fetch_california_housing()
X, y = boston.data, boston.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.1), input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),  # Dropout layer to reduce overfitting
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.01)),  # Elastic Net
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

training_results = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32, verbose=0)

loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f'Test MAE: {mae:.4f}')



def build_model(hp):
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Dense(
        units=hp.Int('units_1', min_value=32, max_value=256, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Choice('l2_1', [0.001, 0.01, 0.1])),
        input_shape=(X_train.shape[1],)
    ))
    
    model.add(layers.Dropout(hp.Choice('dropout_1', [0.2, 0.3, 0.4])))
    
    # Second hidden layer
    model.add(layers.Dense(
        units=hp.Int('units_2', min_value=32, max_value=256, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Choice('l2_2', [0.001, 0.01, 0.1]))
    ))
    model.add(layers.Dropout(hp.Choice('dropout_2', [0.2, 0.3, 0.4])))
    
    # Output layer
    model.add(layers.Dense(1))  
    
    # Compile model with tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 0.01, 0.1])
        ),
        loss='mse',
        metrics=['mae']
    )
    
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_mae', 
    max_epochs=50,
    factor=3,
    directory='kt_tuning',
    project_name='california_housing'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=1)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.hypermodel.build(best_hps)
training_results = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32, verbose=1)

loss, mae = best_model.evaluate(X_test, y_test, verbose=1)
print(f'Optimized Test MAE: {mae:.4f}')


print("Best Hyperparameters:")
print(f"Units in first hidden layer: {best_hps.get('units_1')}")
print(f"L2 regularization for first layer: {best_hps.get('l2_1')}")
print(f"Dropout rate for first layer: {best_hps.get('dropout_1')}")
print(f"Units in second hidden layer: {best_hps.get('units_2')}")
print(f"L2 regularization for second layer: {best_hps.get('l2_2')}")
print(f"Dropout rate for second layer: {best_hps.get('dropout_2')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")
