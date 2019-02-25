import numpy as np
from sklearn.utils import shuffle

final_arrs_x = []
final_arrs_y = []

print("Started Doing Data")

for i in range(0, 9):
    data = np.load("data/data" + str(i) + ".npz")
    data_x = data["x"]
    data_y = data["y"]
    del data

    final_arrs_y.append(data_y)
    ongoing = [data_x[0] for i in range(10)]
    x_to_append = []
    for d in data_x:
        ongoing.pop(0)
        ongoing.append(d)
        x_to_append.append(ongoing)
    final_arrs_x.append(np.array(x_to_append))
    print("Done with array " + str(i))

print("Done merging and modyfing arrays")

#final_arrs_x = np.array(final_arrs_x)
#final_arrs_y = np.array(final_arrs_y)

final_arrs_x_ = np.array(final_arrs_x[0])
final_arrs_y_ = np.array(final_arrs_y[0])

for idx, i in enumerate(final_arrs_x):
    if not idx == 0:
        np.concatenate((final_arrs_x_, i), 0)
        final_arrs_x.pop(0)

for idx, i in enumerate(final_arrs_y):
    if not idx == 0:
        np.concatenate((final_arrs_y_, i), 0)
        final_arrs_y.pop(0)

del final_arrs_x
del final_arrs_y

print(final_arrs_x_.shape)
print(final_arrs_y_.shape)

print("Done Reshaping Arrays")

final_arrs_x_, final_arrs_y_ = shuffle(final_arrs_x_, final_arrs_y_)

print("Done shuffling arrays")

np.savez_compressed("data/data", x=final_arrs_x_, y=final_arrs_y_)

print("Data Saved!")