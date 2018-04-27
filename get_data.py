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
num_files = 12
cv2.imshow("Recorded", np.zeros((300, 400, 3), np.int32))

while True:
    axes = controller.gamepad
    btns = controller.buttons
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
    img = cv2.resize(img, (400, 300))
    keys = key_check()
    cv2.imshow("Live", img)
    if "Y" in btns:
        cv2.imshow("Recorded", img)
        joyKeys = [thumb_lx, trigger]

        data.append([img, joyKeys])

        counter = counter + 1
        print(counter)
    if(data == []):
        cv2.imshow("Recorded", np.zeros((300, 400, 3), np.int32))
    cv2.waitKey(1)
    keys = key_check()
    btns = controller.buttons
    if "DPAD_DOWN" in btns:
        print("Saving!")
        time.sleep(0.5)
        num_files = num_files + 1
        np.save("data/data" + str(num_files) + ".npy", data)
        data = []
        print("Saved! You can now continue.")