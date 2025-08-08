from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

# Initialize SVM model
svm = SVC()

# Perform grid search
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best parameters and scores
best_params = grid_search.best_params_
best_score = grid_search.best_score_
scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

# Plot results
plt.figure(figsize=(10, 6))
plt.bar(range(len(scores)), scores)
plt.xticks(range(len(scores)), [str(param) for param in params], rotation=90)
plt.xlabel('Parameters')
plt.ylabel('Score')
plt.title('Tuning SVM to perfection')
plt.show()

print("Best Parameters:", best_params)
print("Best Score:", best_score)