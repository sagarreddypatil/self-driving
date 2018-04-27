import numpy as np
import cv2
import pyxinput
import tensorflow as tf

print("Imports Done!")

MODEL_PATH = "./model/frozen_conv_model.pb"

with tf.gfile.GFile(MODEL_PATH, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name="")

print("Loaded Model!")

input_tensor = graph.get_tensor_by_name("input_tensor:0")
output = graph.get_tensor_by_name("dense/Tanh:0")

sess = tf.InteractiveSession(graph=graph)

print("Started Converting")

for b in range(1, 12+1):
    fullData = []
    data = np.load("data/data" + str(b) + ".npy")
    prevFrames = []
    for i in data:
        img = i[0]
        axes = i[1]
        img = cv2.resize(img, (200, 150))
        dat = sess.run(output, feed_dict={input_tensor:np.array([img])})
        prevFrames.append(dat)
        if len(prevFrames) > 10:
            prevFrames.pop(0)
        fullData.append([prevFrames, axes])
        #print("Done")
    print(np.array(fullData).shape)
    print("Done file " + str(b))
    np.save('data/finalLSTM.npy', fullData)