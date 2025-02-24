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
