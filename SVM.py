# Support vector classifier 
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier 

from sklearn.svm import SVC

iris = load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

svm_clf = SVC(kernel = 'rbf', C= 1.0)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

score = svm_clf.score(X_test, y_test)
print(f"Accuracy of the SVM Classifier: {score * 100:.2f}%")

#### Hyperparameter Tuning #### 

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],  
    'degree': [1, 2, 3, 4,  5]          
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")


y_pred = best_model.predict(X_test)
print(f"Optimized Model Accuracy: {best_model.score(X_test, y_test) * 100:.2f}%")

print("Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))



# support vector regressor 
from sklearn.datasets import fetch_california_housing

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVR #Support Vector Machine regressor

import numpy as np

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the SVR model
svr_classifier = SVR(kernel='rbf', C=1.0)
svr_classifier.fit(X_train_scaled, y_train)

y_pred = svr_classifier.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

round_mse = round(mse, 3)
round_r2 = round(r2, 3)

print("Mean Squared Error (MSE):", round_mse)
print("R-squared (R2):", round_r2)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale',0.1,1],
    'degree': [2, 3, 4]
}

# Create the SVR model
svr = SVR()

# Perform grid search
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Error analysis
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

round_mse = round(mse, 3)
round_rmse = round(rmse, 3)
round_r2 = round(r2, 3)

print("Best Parameters:", grid_search.best_params_)
print("Mean Squared Error (MSE):", round_mse)
print("Root Mean Squared Error (RMSE):", round_rmse)
print("R-squared (R2):", round_r2)
