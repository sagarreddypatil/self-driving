#------------------------------------------------------MODEL START------------------------------------------------------#

import tensorflow as tf

"""
NVIDIA model used
Image normalization to avoid saturation and make gradients work better.
Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
Drop out (0.5)
Fully connected: neurons: 100, activation: ELU
Fully connected: neurons: 50, activation: ELU
Fully connected: neurons: 10, activation: ELU
Fully connected: neurons: 1 (output)
# the convolution layers are meant to handle feature engineering
the fully connected layer for predicting the steering angle.
dropout avoids overfitting
ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
"""
"""
#TODO: REDUCE LEARNING RATE
def make_model_cnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(24, (5, 5), (2, 2), activation='elu', input_shape=(300, 400, 3)))
    model.add(tf.keras.layers.Conv2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(tf.keras.layers.Conv2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), activation='elu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), activation='elu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='elu'))

    return model

"""

model_input = tf.keras.layers.Input((10, 300, 400, 3))

cnn_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling=None, input_shape=(300, 400, 3))

model = tf.keras.layers.TimeDistributed(cnn_model)(model_input)
model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)
model = tf.keras.layers.CuDNNLSTM(128, return_sequences=False)(model)
model = tf.keras.layers.Dense(2)(model)

final_model = tf.keras.Model(inputs=model_input, outputs=model)

final_model.compile(loss='mse', metrics=['mse', 'mae'], optimizer=tf.keras.optimizers.Adam(lr=1e-5))
final_model.summary()

#------------------------------------------------------MODEL DONE------------------------------------------------------#

print("MODEL MADE")

import numpy as np

X = np.load("data/datas1.npz")['x']
y = np.load("data/datas1.npz")['y']

print("Done array 1")

for i in range(2, 4):
    X = np.concatenate((X, np.load("data/datas" + str(i) + ".npz")['x']))
    y = np.concatenate((y, np.load("data/datas" + str(i) + ".npz")['y']))
    print("Done array {}".format(i))

from sklearn.utils import shuffle
X, y = shuffle(X, y)

print("DATA LOADED")

tfboard = tf.keras.callbacks.TensorBoard(update_freq='batch')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./model_keras/model-{epoch:02d}.hdf5", monitor='loss', mode='min', verbose=1, save_best_only=True)

final_model.fit(X, y, batch_size=42, epochs=100, callbacks=[tfboard, checkpoint])