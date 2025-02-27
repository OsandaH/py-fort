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


