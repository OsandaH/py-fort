## Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


## Split

from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston" # read the data

raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



## Linear Regression

from sklearn.linear_model import LinearRegression

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


# Regularization

## Lasso Regression L1

from sklearn.linear_model import Lasso

alpha_value = 0.02 ##[0.01,0.02,0.05,0.1,0.11,0.5,0.9,1,2]

lasso_reg = Lasso(alpha = alpha_value)
lasso_reg.fit(X_train,y_train)

# predictions on the training data
y_train_pred = lasso_reg.predict(X_train)

mse_train_L1 = mean_squared_error(y_train, y_train_pred)
print(mse_train_L1)

## Ridge Regression L2

from sklearn.linear_model import Ridge

alpha_value = 0.01 ##[0.01,0.02,0.05,0.1,0.11,0.5,0.9,1,2]

ridge_reg = Ridge(alpha = alpha_value)
ridge_reg.fit(X_train,y_train)

# predictions on the training data

y_train_pred = ridge_reg.predict(X_train)

mse_train_L2 = mean_squared_error(y_train, y_train_pred)
print(f"MSE on Training Data for Ridge Regression: {mse_train_L2:.4f}")


## Elastic Regrssion

from sklearn.linear_model import ElasticNet

alpha_val = 0.01
l1_ratio_val  = 0.5 ##(0 -> 1) 

elastic_reg = ElasticNet(alpha = alpha_val, l1_ratio = l1_ratio_val)
elastic_reg.fit(X_train,y_train)

y_train_pred = elastic_reg.predict(X_train)

mse_train_elastic = mean_squared_error(y_train, y_train_pred)

print(mse_train_elastic)

## K-Fold cross validation with lasso regression

X = data
y = target

from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True, random_state=42)

model = Lasso(alpha=0.005)  

mse_scores = []
r2_scores = []

# Perform 3-Fold Cross-Validation
for train_index, val_index in kf.split(X):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    mse_scores.append(mean_squared_error(y_val, y_pred))
    r2_scores.append(r2_score(y_val, y_pred))

mse_avg = np.mean(mse_scores)
r2_avg = np.mean(r2_scores)

print(f"Average MSE: {mse_avg:.4f}")
print(f"Average R^2: {r2_avg:.4f}")



## Data Analysis 

# import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv") # read the dataset

df.head(10) # 1st 10 lines of the dataset

df.shape # rows and columns 

df.shape[0] # rows 
df.shape[1] # columns 

# adding header Row 
column_headers = [
    "symboling","normalized_losses","make","fuel_type",
    "aspiration","num_doors","body_style","drive_wheels",
    "engine_location","wheel_base","length","width","height",
    "curb_weight","engine_type","num_cylinders","engine_size",
    "fuel_system","bore","stroke","compression_ratio","horsepower",
    "peak_rpm","city_mpg","highway_mpg","price"]

df = pd.read_csv("dataset.csv", names=column_headers, na_values=["?"])

# number of null values
#df.isna().sum()
null_values = df.isna()
num_nulls = null_values.sum()
#print(num_nulls)

# number of duplicates 
duplicates = df.duplicated()
num_dups = duplicates.sum()
#print("\nNumber of duplicates: ",num_dups)

#print the duplicates
duplicates = df.duplicated(keep = False)
dup_rows = df[duplicates]
#print(dup_rows)

# Drop the Duplicates 
df_unique = df.drop_duplicates() 

# Drop missing values 
df_new = df_unique.dropna()


#### numerical data ####

numerical_cols = df.select_dtypes(include = 'number').columns 

# adding mean value to missing cols 
df_mean = df_new.copy()
df_mean[numerical_cols] = df_mean[numerical_cols].fillna(df_mean[numerical_cols].mean())

# fill the missing values with median
df_median = df_new.copy()
df_median[numerical_cols] = df_median[numerical_cols].fillna(df_new[numerical_cols].median)

#### categorical data ####

string_column = df_mf.select_dtypes(include='object').columns

# adding mean value to missing cols
df_mf = df_mean.copy()
for column in string_column:
    df_mf[column].fillna(df_mf[column].mode()[0], inplace = True)

# fill the missing values with median
df_mef = df_median.copy()
string_column = df_mef.select_dtypes(include='object').columns
for column in string_column:
    df_mef[column].fillna(df_mef[column].mode()[0], inplace = True)
    
    
#### One Hot Encoding ####

categorical_columns = df_new.select_dtypes(include = 'object').columns
df_encoded = pd.get_dummies(df_new, columns = categorical_columns, dtype = int)

#### statistics ####
numericalData = df_mean.select_dtypes(include = 'number')

stats = pd.DataFrame({
    'Mean': numericalData.mean(),
    'Median': numericalData.median(),
    'Mode': numericalData.mode().iloc[0],  # Selecting the first mode
    'Variance': numericalData.var(),
    'Standard Deviation': numericalData.std(),
    'Minimum': numericalData.min(),
    'Maximum': numericalData.max()
})
## print(stats)

# ## data types
# df.info()

# ## description
# df.describe()

## selecting few columns 
df1 = df[['normalized_losses','symboling']]

## group by
df[['make','horsepower']].groupby('make').mean()

##
mean_val = df['horsepower'].mean()
mode_values = df['horsepower'].mode().iloc[0] 
median_values = df['horsepower'].median()


## Correlation Matrix

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet 
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("car_data.csv")

## drop a column 
df_new = df.drop(columns = ["car_name"])

## data types of columns 
df_new.dtypes

## Rows with null values
null_rows = df_new[df_new.isnull().any(axis=1)]

## filling missing values with mean 
df_mean_fill = df_new.copy()
num_cols = df_new.select_dtypes(include='number').columns
df_mean_fill[num_cols] = df_mean_fill[num_cols].fillna(df_mean_fill[num_cols].mean())

df_1 = df_mean_fill.copy()
## describe 1 element in each column 
std = df.describe().loc["std"]
maxSTD = np.max(std)

correlation_matrix = df_1.corr()
#print(correlation_matrix)
plt.figure(figsize = (10,8))
plt.imshow(correlation_matrix, cmap = "coolwarm")
plt.colorbar()
plt.xticks(range(len(correlation_matrix)),correlation_matrix.columns,rotation = 90)
plt.xticks(range(len(correlation_matrix)),correlation_matrix.columns)
plt.title('Correlation Matrix Heatmap')
plt.show()

#### Positive pairs ####
positive_pairs_df = correlation_matrix.unstack()
positive_pairs_df = positive_pairs_df[(positive_pairs_df > 0) & (positive_pairs_df.index.get_level_values(0) != positive_pairs_df.index.get_level_values(1))]
positive_pairs_df = positive_pairs_df[positive_pairs_df.index.map(lambda x: x[0] < x[1])]

#### Negative pairs ####
positive_pairs_df_neg = correlation_matrix.unstack() # Unstack the matrix
positive_pairs_df_neg = positive_pairs_df_neg[(positive_pairs_df_neg < 0) & (positive_pairs_df_neg.index.get_level_values(0) != positive_pairs_df_neg.index.get_level_values(1))]
positive_pairs_df_neg = positive_pairs_df_neg[positive_pairs_df_neg.index.map(lambda x: x[0] < x[1])]

#### Pairs Higher than a specific value ####
high_corr_pairs = positive_pairs_df[positive_pairs_df > 0.8]




## Standardization

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

numerical_columns = df_1.select_dtypes(include = 'number').columns
df_numerical = df_1[numerical_columns]

#### Z score scandalization ####
scaler = StandardScaler()
df_scaled_zscore = scaler.fit_transform(df_numerical)
df_scaled_zscore = pd.DataFrame(df_scaled_zscore, columns=numerical_cols)
#print(df_scaled_zscore.head())

#### Min Max Scaling ####
scaler = MinMaxScaler()
df_scaled_MinMax = scaler.fit_transform(df_numerical)
df_scaled_MinMax = pd.DataFrame(df_scaled_MinMax, columns=numerical_cols)
#print(df_scaled_MinMax.head())

#### Robust Scaling ####
scaler = RobustScaler()
df_scaled_robust = scaler.fit_transform(df_numerical)
df_scaled_robust = pd.DataFrame(df_scaled_robust, columns=numerical_cols)
#print(df_scaled_robust.head())

## Decision Tree & Ensemble 

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_iris

from sklearn.model_selection import GridSearchCV


iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#### Plot the decision tree ####

clf =  DecisionTreeClassifier(random_state = 42).fit(X, y)
plt.figure(figsize=(16, 10))
plot_tree(
    clf,
    feature_names=iris.feature_names,  
    class_names=iris.target_names,    
    filled=True                      
)
plt.title("Decision Tree Trained on All Iris Features") 
plt.show()


#### depth of the tree ####
depth = clf.get_depth()

print(f'Depth: {depth}')

y_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(conf_matrix)

clsf_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification Report")
print(clsf_report)



##### AdaBoost ####
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_boost.fit(X_train,y_train)

y_pred = ada_boost.predict(X_test)
score = ada_boost.score(X_test, y_test)
print(f'Accuracy of the AdaBoost model: {score*100:.2f}%')


#### Gradient Boosting ####

Gradient_Boosting_clf = GradientBoostingClassifier(n_estimators=100, 
                                                  learning_rate=0.1, 
                                                  max_depth=3, 
                                                  random_state=42)
Gradient_Boosting_clf.fit(X_train, y_train)
y_pred_Grad = Gradient_Boosting_clf.predict(X_test)
score = Gradient_Boosting_clf.score(X_test, y_test) 
print(f"Accuracy of Gradient Boosting Model: {score * 100:.2f}%")


#### RandomForest ####
RF_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
RF_clf.fit(X_train, y_train)

y_pred = RF_clf.predict(X_test)

score = RF_clf.score(X_test, y_test)
print(f"Accuracy of the Random Forest model: {score * 100:.2f}%")


##### AdaBoost Hyperparaeter Tuning ####


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 1.0, 1.5],
    'estimator__max_depth': [1, 2, 3, 4, 5]  # Use 'estimator__' instead of 'base_estimator__'
}

# Create base estimator
base_estimator = DecisionTreeClassifier()

# Initialize AdaBoost with base estimator
ada_boost = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(ada_boost, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Perform the search
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

# Print results
print(f"Best hyperparameters: {best_params}")
print(f"Best cross-validated accuracy: {best_score * 100:.2f}%")

# Evaluate on test data
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


#### Gradient Boosting hyperparameter Tuning ####


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 1.0, 1.5],
    'max_depth': [1, 2, 3, 4, 5]  # Corrected parameter name
}

# Initialize Gradient Boosting Classifier
grad_boost = GradientBoostingClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(grad_boost, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Perform the search
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

# Print results
print(f"Best hyperparameters: {best_params}")
print(f"Best cross-validated accuracy: {best_score * 100:.2f}%")

# Evaluate on test data
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


#### RandomForest Hyperparameter tuning ####

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

# Perform the search
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

# Print results
print(f"Best hyperparameters: {best_params}")
print(f"Best cross-validated accuracy: {best_score * 100:.2f}%")

# Evaluate on test data
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


## SVC

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


## SVR

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


## DNN

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


## Unsupervised Learning

#### K-means Clustering ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k_values = [2, 3, 4, 5, 6, 7, 8]

# Store silhouette scores
silhouette_scores = []
inertia_values = []

# Perform K-means clustering for different values of k
plt.figure(figsize=(18, 12))
for i, k in enumerate(k_values, 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X, labels))
    inertia_values.append(kmeans.inertia_)
    
    plt.subplot(3, 3, i)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50, alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    
plt.tight_layout()
plt.show()

# Find optimal k using Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score is: k={optimal_k}")


# Find optimal k using Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='r')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.show()

optimal_k = k_values[np.diff(inertia_values, 2).argmin() + 1]
print(f"\nOptimal number of clusters based on Elbow Method is: k={optimal_k}")




df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k_values = [2, 3, 4, 5, 6, 7, 8]

# Store silhouette scores
silhouette_scores = []
log_likelihood_values = []

plt.figure(figsize=(18, 12))
for i, k in enumerate(k_values, 1):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X)
    centers = gmm.means_
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X, labels))
    log_likelihood_values.append(gmm.score(X) * len(X))
    
    plt.subplot(3, 3, i)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    plt.title(f'GMM Clustering (k={k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    
plt.tight_layout()
plt.show()

# Find optimal k using Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score (GMM) is: k={optimal_k_silhouette}")


# Plot log-likelihood vs. number of clusters
plt.figure(figsize=(8, 5))
plt.plot(k_values, log_likelihood_values, marker='o', linestyle='--', color='purple')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Number of Clusters')
plt.show()

optimal_k_likelihood = k_values[np.argmax(log_likelihood_values)]
print(f"\nOptimal number of clusters based on Log-Likelihood (GMM) is: k={optimal_k_likelihood}")




# dentify the three clusters in Iris dataset using k-means clustering and use  mutual information (MI) to validate the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y_true = iris.target

# number of clusters
n_clusters = 3

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_mi = mutual_info_score(y_true, kmeans_labels)

print(f'K-Means Mutual Information Score: {kmeans_mi:.4f}')

# GMM Clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm_labels = gmm.fit_predict(X)
gmm_mi = mutual_info_score(y_true, gmm_labels)

print(f'GMM Mutual Information Score: {gmm_mi:.4f}')

## CNN

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


