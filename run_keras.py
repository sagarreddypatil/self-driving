import tensorflow as tf
import pyxinput
from mss import mss
import cv2
import numpy as np
import time

joystick = pyxinput.vController()
MODEL_PATH = "./model_keras/model-42.hdf5"

model = tf.keras.models.load_model(MODEL_PATH)

sct = mss()
mon = {'top': 31, 'left': 1, 'width': 800, 'height': 600}

img = np.array(sct.grab(mon))
cv2.imshow("Seen", img)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
img = cv2.resize(img, (400, 300))

imgs = [img for i in range(10)]

print("STARTING YAAY")
while True:
    img = np.array(sct.grab(mon))
    cv2.imshow("Seen", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.resize(img, (400, 300))

    imgs.pop(0)
    imgs.append(img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    ans = model.predict(np.array([imgs]))

    print(ans[0])

    xa = ans[0][0]
    ya = ans[0][1]

    if xa > 1:
        xa = 1
    if xa < -1:
        xa = -1
    
    if ya > 1:
        ya = 1
    if ya < -1:
        ya = -1
    joystick.set_value("AxisLx", xa)
    joystick.set_value("AxisLy", ya)