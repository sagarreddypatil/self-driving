import pyxinput
import cv2
from mss import mss
import numpy as np
from getKeys import key_check
from utils import apply_deadzone
import time

controller = pyxinput.rController(1)
sct = mss()
mon = {'top': 38, 'left': 1, 'width': 800, 'height': 600}

data = []

counter = 1
num_files = 12

while True:
    img = np.array(sct.grab(mon)) #converting PIL image to numpy array. For CSV Panda's frame is better. FOr image Numpy array is better
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #converting BGRA color format to RGB. Identified it as BGRA color as I was getting 4D data from Numpy array
    cv2.imshow("Live", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break