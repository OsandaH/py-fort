## Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


## Split

data_url = "http://lib.stat.cmu.edu/datasets/boston" # read the data

raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



## Linear Regression

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

train_pred = model.predict(X_train)

mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared value

train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, train_pred)  # Training MSE
mse_test = mean_squared_error(y_test, y_pred)  # Testing MSE


print(f"Training Mean Squared Error (MSE): {mse_train:.4f}")
print(f"Testing Mean Squared Error (MSE): {mse_test:.4f}")


## Scatter data 

plt.scatter(y_test, y_pred)
plt.xlabel("y_test (Testing Values)")
plt.ylabel("y_pred (Predicted Values)")
plt.title("y_pred vs y_test")
plt.show()

