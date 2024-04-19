from sklearn import datasets
from sklearn import svm

import matplotlib as plt
from sklearn.model_selection import train_test_split

def show_four_digits_from_dataset ():
  _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
  for ax, image, label in zip(axes, digits.images, digits.targets):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Target: %i" % label)

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split()

x = -10
clf.fit(digits.data[:x], digits.target[:x])
print('Predicted Data:','\n', clf.predict(digits.data[x:]))
print('Target Data:\n', digits.target[x:])