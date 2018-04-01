import tensorflow as tf
import pyxinput
from mss import mss
import cv2
import numpy as np
import time


sct = mss()
mon = {'top': 38, 'left': 0, 'width': 800, 'height': 600}
#saver = tf.train.import_meta_graph("C:\\Users\\SagarVeeru\\Documents\\SagarStuff\\Code\\Python\\self-driving\\model\\model-43.meta")
print("STARTING YAAY")
while True:
    img = np.array(sct.grab(mon))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imshow('lol', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break