import matplotlib.pyplot as plt
from sklearn import datasets, svm

# importing dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)


# ------------------------------------
# ------------ SVC -------------------
# ------------------------------------
svc = svm.SVC()
svc.fit(iris_X, iris_y)
predicted = svc.predict(iris_X)

# --------------- Learning Curve -----
from sklearn.model_selection import LearningCurveDisplay, learning_curve

train_sizes, train_scores, test_scores = learning_curve(svc, iris_X, iris_y)
display_learning_curve = LearningCurveDisplay(
  train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores
)
display_learning_curve.plot()
plt.title('SVC - Learning Curve')
plt.show()

# ------------ Error -----------------
from sklearn.metrics import PredictionErrorDisplay

display_error = PredictionErrorDisplay(y_true=iris_y, y_pred=predicted)
display_error.plot()
plt.title('SVC - Prediction Error')
plt.show()

# ------------------------------------
# ----- Decision Tree ----------------
# ------------------------------------
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=4)
tree.fit(iris_X, iris_y)
predicted = tree.predict(iris_X)

# -------- Learning Curve ------------
train_sizes, train_scores, test_scores = learning_curve(tree, iris_X, iris_y)
display_learning_curve = LearningCurveDisplay(
  train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores
)
display_learning_curve.plot()
plt.title('Decision Tree - Learning Curve')
plt.show()

# ------------ Error -----------------
display_error = PredictionErrorDisplay(y_true=iris_y, y_pred=predicted)
display_error.plot()
plt.title('Decision Tree - Prediction Error')
plt.show()

# ---------------------------------------------------
# ----------- Ridge ---------------------------------
# ---------------------------------------------------
from sklearn.linear_model import Ridge

ridge = Ridge().fit(iris_X, iris_y)
predicted = ridge.predict(iris_X)

# -------- Learning Curve ------------
train_sizes, train_scores, test_scores = learning_curve(ridge, iris_X, iris_y)
display_learning_curve = LearningCurveDisplay(
  train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores
)
display_learning_curve.plot()
plt.title('Ridge - Learning Curve')
plt.show()

# ------------ Error -----------------
display_error = PredictionErrorDisplay(y_true=iris_y, y_pred=predicted)
display_error.plot()
plt.title('Ridge - Prediction Error')
plt.show()


# ---------------------------------------------------
# ----------- MLP -----------------------
# ---------------------------------------------------
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=1, max_iter=300).fit(iris_X, iris_y)
predicted = mlp.predict(iris_X)

# -------- Learning Curve ------------
train_sizes, train_scores, test_scores = learning_curve(mlp, iris_X, iris_y)
display_learning_curve = LearningCurveDisplay(
  train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores
)
display_learning_curve.plot()
plt.title('MLP - Learning Curve')
plt.show()

# ------------ Error -----------------
display_error = PredictionErrorDisplay(y_true=iris_y, y_pred=predicted)
display_error.plot()
plt.title('MLP - Prediction Error')
plt.show()