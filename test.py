import numpy as np
import cv2

for b in range(1, 6):
    data = np.load("data/data" + str(b) + ".npy")
    for i in data:
        cv2.imshow('wow', i[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break