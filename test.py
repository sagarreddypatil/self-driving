import numpy as np


X = np.load("data/datas1.npz")['x']
y = np.load("data/datas1.npz")['y']

print("Done array 1")

for i in range(2, 9):
    X = np.concatenate((X, np.load("data/datas" + str(i) + ".npz")['x']))
    y = np.concatenate((y, np.load("data/datas" + str(i) + ".npz")['y']))
    print("Done array {}".format(i))

from sklearn.utils import shuffle
X, y = shuffle(X, y)