import pyxinput
import cv2
from mss import mss
import numpy as np
from getKeys import key_check
from utils import apply_deadzone
import time

controller = pyxinput.rController(1)
sct = mss()
mon = {'top': 38, 'left': 0, 'width': 800, 'height': 600}

data = []

counter = 1
num_files = 0

paused = True

while True:
    axes = controller.gamepad
    thumb_lx = axes['thumb_lx']
    thumb_lx = float(thumb_lx) / 32768.0
    thumb_lx = float("%5.3f" % thumb_lx)

    lt = float(axes['left_trigger'])
    rt = float(axes['right_trigger'])

    trigger = (rt - lt) / 255.0
    trigger = float("%5.3f" % trigger)

    thumb_lx = apply_deadzone(thumb_lx)
    trigger = apply_deadzone(trigger)

    img = np.array(sct.grab(mon))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.resize(img, (200, 150))
    if not paused:
        joyKeys = [thumb_lx, trigger]

        data.append([img, joyKeys])

        counter = counter + 1
        
    keys = key_check()
    if "L" in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(0.4)
        else:
            print('Pausing!')
            paused = True
            time.sleep(0.5)
            num_files = num_files + 1
            np.save("data/data" + str(num_files) + ".npy", data)
            data = []
    print(counter)