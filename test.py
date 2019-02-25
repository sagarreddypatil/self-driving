import cv2
from mss import mss
import numpy as np

cap = cv2.VideoCapture(0)
sct = mss()
mon = {'top': 31, 'left': 1, 'width': 800, 'height': 600}

while True:
    img = np.array(sct.grab(mon))
    cv2.imshow("yeah", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break