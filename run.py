import tensorflow as tf
import pyxinput
from mss import mss
import cv2
import numpy as np
import time

joystick = pyxinput.vController()
MODEL_PATH = "./model/frozen_model.pb"

with tf.gfile.GFile(MODEL_PATH, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name="")

input_tensor = graph.get_tensor_by_name("input_tensor:0")
output = graph.get_tensor_by_name("output:0")

sct = mss()
mon = {'top': 38, 'left': 0, 'width': 800, 'height': 600}
with tf.Session(graph=graph) as sess:
    print("STARTING YAAY")
    while True:
        img = np.array(sct.grab(mon))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (200, 150))

        ans = sess.run(output, feed_dict={input_tensor: np.array([img])})

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