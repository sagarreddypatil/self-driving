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

model_input = tf.keras.layers.Input((10, 300, 400, 3))
model = tf.keras.layers.TimeDistributed(make_model_cnn())(model_input)
model = tf.keras.layers.CuDNNLSTM(256, return_sequences=False)(model)
model = tf.keras.layers.Dense(2, activation='tanh')(model)

final_model = tf.keras.Model(inputs=model_input, outputs=model)

final_model.compile(loss='mse', metrics=['mse', 'mae', 'mape'], optimizer=tf.keras.optimizers.Adam(lr=1e-4))

#------------------------------------------------------MODEL DONE------------------------------------------------------#

print("MODEL MADE")

import numpy as np

data = np.load("data/data.npz")

data_x = data['x']
data_y = data['y']
del data

print("DATA LOADED")

tfboard = tf.keras.callbacks.TensorBoard(update_freq='batch')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./model_keras/model.hdf5", monitor='loss')

final_model.fit(data_x, data_y, batch_size=42, epochs=100, callbacks=[tfboard, checkpoint])