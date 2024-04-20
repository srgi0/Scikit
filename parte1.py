# ------------------ Functions --------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

def show_n_digits_from_dataset (base_digit, n_digits):
  _, axes = plt.subplots(nrows=1, ncols=n_digits, figsize=(10,3))
  for ax, image, label in \
  zip(axes, digits.images[base_digit:base_digit+n_digits], digits.target[base_digit:base_digit+n_digits]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Target: %i" % label)
  plt.show()

show_n_digits_from_dataset(base_digit=0,n_digits=5)

# -----------------------------------------------
# ----- SVC ---------------------------------------------------------------------------------------------------
# -----------------------------------------------

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

X_train, X_test, y_train, y_test = train_test_split(
  data, digits.target, test_size=0.5, shuffle=False
)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# ------- Learning Curve -------------------
from sklearn.model_selection import LearningCurveDisplay, learning_curve

train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train)
display_learning_curve = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores)
display_learning_curve.plot()
plt.title('SVC - Learning Curve')
plt.show()

# ------- Error ----------------------------
from sklearn.metrics import PredictionErrorDisplay

display_error = PredictionErrorDisplay(y_true=y_test, y_pred=predicted)
display_error.plot()
plt.title('SVC - Prediction Error')
plt.show()

# --------------------------------------------------------------------------------------------------------------
# ----------- Decision Tree ------------------------------------------------------
# -------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_digits(return_X_y=True)
tree = DecisionTreeClassifier(random_state=4)
tree.fit(X, y)
predicted = tree.predict(X)

# -------- Learning Curve -------------------------------
train_sizes, train_scores, test_scores = learning_curve(tree, X, y)
display_learning_curve = LearningCurveDisplay(train_sizes=train_sizes,
  train_scores=train_scores, test_scores=test_scores, score_name="Score")
display_learning_curve.plot()
plt.title('Decision Tree - Learning Curve')
plt.show()

# ------------ Error ------------------------------------
display_error = PredictionErrorDisplay(y_true=y, y_pred=predicted)
display_error.plot()
plt.title('Decision Tree - Prediction Error')
plt.show()

# ---------------------------------------------------
# ----------- Ridge ---------------------------------
# ---------------------------------------------------
from sklearn.linear_model import Ridge

X, y = datasets.load_diabetes(return_X_y=True)
ridge = Ridge().fit(X, y)
y_pred = ridge.predict(X)

# -------- Learning Curve -------------------------------
train_sizes, train_scores, test_scores = learning_curve(ridge, X, y)
display_learning_curve = LearningCurveDisplay(train_sizes=train_sizes,
  train_scores=train_scores, test_scores=test_scores, score_name="Score")
display_learning_curve.plot()
plt.title('Ridge - Learning Curve')
plt.show()

# ------------ Error --------------------------------
display_error = PredictionErrorDisplay(y_true=y, y_pred=y_pred)
display_error.plot()
plt.title('Ridge - Prediction Error')
plt.show()