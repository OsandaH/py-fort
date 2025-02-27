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



## K-Fold cross validation with lasso regression ##
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

###### hy tu ### 
 from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

# Define parameter grids
lasso_params = {'alpha': [0.01, 0.02, 0.05, 0.1, 0.5, 1, 2]}
ridge_params = {'alpha': [0.01, 0.02, 0.05, 0.1, 0.5, 1, 2]}
elastic_params = {'alpha': [0.01, 0.02, 0.05, 0.1, 0.5, 1, 2], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

# Perform Grid Search
lasso_grid = GridSearchCV(Lasso(), lasso_params, scoring='neg_mean_squared_error', cv=5)
ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='neg_mean_squared_error', cv=5)
elastic_grid = GridSearchCV(ElasticNet(), elastic_params, scoring='neg_mean_squared_error', cv=5)

# Fit models
lasso_grid.fit(X_train, y_train)
ridge_grid.fit(X_train, y_train)
elastic_grid.fit(X_train, y_train)

# Best models
best_lasso = lasso_grid.best_estimator_
best_ridge = ridge_grid.best_estimator_
best_elastic = elastic_grid.best_estimator_

# Evaluate models on training data
y_train_pred_lasso = best_lasso.predict(X_train)
y_train_pred_ridge = best_ridge.predict(X_train)
y_train_pred_elastic = best_elastic.predict(X_train)

mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
mse_elastic = mean_squared_error(y_train, y_train_pred_elastic)

# Print results
print(f"Best Lasso Alpha: {lasso_grid.best_params_['alpha']}, MSE: {mse_lasso:.4f}")
print(f"Best Ridge Alpha: {ridge_grid.best_params_['alpha']}, MSE: {mse_ridge:.4f}")
print(f"Best ElasticNet Alpha: {elastic_grid.best_params_['alpha']}, L1 Ratio: {elastic_grid.best_params_['l1_ratio']}, MSE: {mse_elastic:.4f}")

# Find the best model based on lowest MSE
mse_dict = {'Lasso': mse_lasso, 'Ridge': mse_ridge, 'Elastic Net': mse_elastic}
best_model_name = min(mse_dict, key=mse_dict.get)
print(f"Best model: {best_model_name} with MSE: {mse_dict[best_model_name]:.4f}")




