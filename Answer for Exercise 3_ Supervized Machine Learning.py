###################################################
""" Code written for the Exercise 2 Supervised Machine Learning of the Python and Machine Learning.
     Written By Worku Biniyam
     Submitted to Michael E. Rose (PhD)
     Date of submission date August 30, 2022.
"""
# ------------------------- QUESTION ONE---------------
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
# ------------------------------------------------------------
# 1.a loading the dataset from csv File
df = pd.read_csv('./output/polynomials.csv', sep=',', index_col=0)
# ------------------------------------------------------------
# 1.b Splitting the DataFrame into data and target

y = df["Y"]
X = df.drop(columns='Y')

# ------------------------------------------------------------
# 1.c Running the Ordinary OLS Model

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lm = OLS().fit(X_train, y_train)
lm.score(X_test, y_test)

ridge = Ridge(alpha=0.3).fit(X_train, y_train)  # Train the model using Ridge.
ridge.score(X_test, y_test)
lasso = Lasso(alpha=0.3).fit(X_train, y_train)  # Train the model using LASSO.
lasso.score(X_test, y_test)

# ------------------------------------
# 1.d Creating Pandas DataFrame
coefficients = pd.DataFrame(lm.coef_, index=X.columns, columns=["OLS"])

coefficients["Ridge"] = ridge.coef_
coefficients["Lasso"] = lasso.coef_

sum((coefficients["Lasso"] == 0) & (coefficients["Ridge"] != 0))
# There are 17 rows where the Lasso coefficients equals to zero and while Ridge coefficients if different from Zero.
# ------------------------------------------------------------
# 1.e Plotting bar figure and saving
coefficients = coefficients.plot.barh(figsize=(10, 30))
coefficients.figure.savefig("./output/polynomials.pdf")

# ------------------------------------------------------------
# Neural Network Regression
# 2.a loading the dataset

diabetes = load_diabetes()
print(diabetes["DESCR"])

X = diabetes["data"]
Y = diabetes["target"]
# ------------------------------------------------------------
# 2.b Creating Neural Network Regression

scaler = StandardScaler().fit(X_train)  # Scaling the dataset using the MinMaxScaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluating our scale that it is between the interval -1 and 1+.
X_train_scaled.max()
X_test_scaled.max()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)


algorithms = [("scaler", StandardScaler()),
              ("nn", MLPRegressor(solver="lbfgs", random_state=42, max_iter=1000))]
pipe = Pipeline(algorithms, verbose=True)
parameter_grid = {"nn__hidden_layer_sizes": [(75, 75), (100, 100), (150, 150)],
                  "nn__alpha": [0.01, 0.005, 0.001]}
grid = GridSearchCV(pipe, parameter_grid, cv=3)
grid.fit(X_train, Y_train)
# ------------------------------------------------------------
# 2.c Model Selection
model = pd.DataFrame(grid.cv.results_)
best_model = grid.best_params_
# The best model is nn__alpha': 0.005 with a number of hidden_layer_sizes of (150, 150).
# ------------------------------------------------------------
# 2.d Plotting heatmap and saving

results_again = pd.DataFrame(grid.cv_results_)
scores_again = results_again["mean_test_score"].values.reshape(3, 3)
coefficient_heatmap = sns.heatmap(data=scores_again, annot=True, xticklabels=parameter_grid["nn__hidden_layer_sizes"],
            yticklabels=parameter_grid["nn__alpha"])
coefficient_heatmap.savefig("./output/nn_diabetes_importances.pdf.")
# ------------------------------------------------------------
# 3.a

cancer = load_breast_cancer()
Y = cancer["target"]
X = cancer["data"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)


algorith = [("scaler", MinMaxScaler()), ("nn", MLPClassifier(solver="lbfgs", random_state=42, max_iter=1000))]
pip = Pipeline(algorith, verbose=True)
param_grd = {"nn__hidden_layer_sizes": [(100, 100), (150, 150)], "nn__alpha": [0.01, 0.1]}
grd = GridSearchCV(pip, param_grd, scoring="roc_auc")
grd.fit(X_train, Y_train)

# 3.c Neural Network Classifier
model_result = pd.DataFrame(grd.cv_results_)
print(best_model=grd.best_estimator_)

#  The best model is at alpha = 0.1 and hidden_layer_size of (150,150) with a ROC-AuC-score of 0.993.

# 3.d Plotting and saving of confusion matrix Heatmap

prediction = grd.predict(X_test)
matrix = confusion_matrix(Y_test, prediction)
heatmap = sns.heatmap(matrix, annot=True, xticklabels=cancer["target_names"], yticklabels=cancer["target_names"])
heatmap.figure.savefig("./output/nn_breast_confusion.pdf")
# --------------------------------- END ------------------------------------------------------------
