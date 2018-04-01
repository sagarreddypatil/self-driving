import numpy as np

final = []
for b in range(1, 10):
    data = np.load("data/data" + str(b) + ".npy")
    if b == 1:
        final = data
    else:
        final = np.concatenate((final, data), axis=0)
    print("Loaded " + str(b))
np.random.shuffle(final)
print("Shuffled")
np.save('data/final.npy', final)
print("Saved")