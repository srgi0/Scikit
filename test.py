import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

def show_four_digits_from_dataset ():
  _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
  for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Target: %i" % label)

print(digits.target[:4])
show_four_digits_from_dataset()