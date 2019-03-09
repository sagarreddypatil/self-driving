import numpy as np
from sklearn.utils import shuffle

final_arrs_x = []
final_arrs_y = []

print("Started Doing Data")

i = 4
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

final_arrs_x = np.concatenate(final_arrs_x, axis=0)
final_arrs_y = np.concatenate(final_arrs_y, axis=0)

print(final_arrs_x.shape)
print(final_arrs_y.shape)

print("Done Reshaping Arrays")

final_arrs_x, final_arrs_y = shuffle(final_arrs_x, final_arrs_y)

print("Done shuffling arrays")

np.savez_compressed("data/datas" + str(i), x=final_arrs_x, y=final_arrs_y)

print("Data Saved!")