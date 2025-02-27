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





