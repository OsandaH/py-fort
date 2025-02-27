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

