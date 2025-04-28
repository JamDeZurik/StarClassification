import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

#Stellar Classification Dataset
#https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
df = pd.read_csv("star_classification.csv").sample(1000)
X = df.iloc[:, :-1].copy().to_numpy() #Features
y = df.iloc[:, -1].copy().to_numpy() #Label

#normalize the data because the magnitudes vary
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

#saving the train test split and classifier for the grid search
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(kernel='rbf', decision_function_shape='ovo')

#grid search to find the best C and gamma parameters
parameters = {"C": np.linspace(10, 100, 10), "gamma": np.linspace(0.001, 0.1, 10)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train) #clf.fit(X_train, y_train)
results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_C', 'mean_test_score', 'rank_test_score']])
print(results[['param_gamma', 'mean_test_score', 'rank_test_score']])

#save the results
grid_C = grid_search.best_params_["C"]
grid_gamma = grid_search.best_params_["gamma"]

print(f"C: {grid_C}, Gamma:{grid_gamma}")

#running support vector classifier (radial basis function) with best parameters
clf = SVC(kernel='rbf', decision_function_shape='ovo', C=grid_C, gamma=grid_gamma)
clf.fit(X_train, y_train)

#NOTE: "score" might not be the optimal choice because the dataset is imbalanced
#(there are many GALAXY entries)
print(f"Score: {clf.score(X_test, y_test):.3f}")

#display results in a confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
