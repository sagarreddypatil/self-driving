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
	tf.layers.dropout avoids overfitting
	ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
"""
 
EPOCHS = 1000
BATCH_SIZE = 64
 
import tensorflow as tf
import numpy as np
 
local_response_normalization = tf.nn.local_response_normalization
conv_2d = tf.layers.conv2d
max_pool_2d = tf.layers.max_pooling2d
fully_connected = tf.layers.dense
dropout = tf.layers.dropout
 
print("Done importing libraries")
 
input_tensor = tf.placeholder(tf.float32, [None, 150, 200, 3], name='input_tensor') # 150x200 has the same aspect ratio as 300x400
y_true = tf.placeholder(tf.float32, [None, 2], name='y_true')
training = tf.placeholder(tf.bool, name='training')
 
model = input_tensor

#TODO: CHANGE THE NORMALISATION FROM -1 TO 1 TO 0 TO 1 BECAUSE ACTIVATION FUNCTION IS RELU AND IT WILL DISCARD ANY NEGATIVE VALUE
model = tf.subtract(tf.divide(model, 127.5), 1.0, name='normalize') #each pixel has range of 0 to 255. This is to scale 0 to 255 to -1 to 1
#Below is from AlexNet model

model = conv_2d(model, 96, 11, strides=4, activation=tf.nn.relu, padding='same')
model = max_pool_2d(model, 3, strides=2, padding='same')
model = local_response_normalization(model)
model = conv_2d(model, 256, 5, activation=tf.nn.relu, padding='same')
model = max_pool_2d(model, 3, strides=2, padding='same')
model = local_response_normalization(model)
model = conv_2d(model, 384, 3, activation=tf.nn.relu, padding='same')
model = conv_2d(model, 384, 3, activation=tf.nn.relu, padding='same')
model = conv_2d(model, 256, 3, activation=tf.nn.relu, padding='same')
model = max_pool_2d(model, 3, strides=2, padding='same')
model = conv_2d(model, 256, 5, activation=tf.nn.relu, padding='same')
model = max_pool_2d(model, 3, strides=2, padding='same')
model = local_response_normalization(model)
model = conv_2d(model, 384, 3, activation=tf.nn.relu, padding='same')
model = conv_2d(model, 384, 3, activation=tf.nn.relu, padding='same')
model = conv_2d(model, 256, 3, activation=tf.nn.relu, padding='same')
model = max_pool_2d(model, 3, strides=2, padding='same')
model = local_response_normalization(model)
model = tf.layers.flatten(model)
model = fully_connected(model, 4096, activation=tf.nn.tanh)
model = dropout(model, 0.5, training)
model = fully_connected(model, 4096, activation=tf.nn.tanh)
model = dropout(model, 0.5, training)
model = fully_connected(model, 4096, activation=tf.nn.tanh)
model = dropout(model, 0.5, training)
model = fully_connected(model, 4096, activation=tf.nn.tanh)
model = dropout(model, 0.5, training)
model = fully_connected(model, 2) #TODO: CHANGE ACTIVATION FUNCTION TO TANH BECAUSE THE VALUES WE NEED IS BETWEEN -1 AND 1 
output = tf.multiply(model, 1.0, name='output') #Just to get an output node for later use
 
print("MODEL DEFINED")
 
loss = tf.reduce_mean(tf.losses.mean_squared_error(y_true, output)) #For Regression Mean Squared Error is better
optimizer = tf.train.AdamOptimizer(0.00007).minimize(loss)
 
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)
tf.summary.scalar('loss', loss) #This is for TensorBoard loss graphs
merged_summary_op = tf.summary.merge_all()
 
data = np.load("data/final.npy")
imgs = []
axes = []
for d in data:
	imgs.append(d[0])
	axes.append(d[1])
del data
print("DATA LOADED")
 
print("STARTING TRAINING")
 
with tf.Session() as sess:
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph()) # This is for Tensor Board
    total_counter = 0
    for a in range(1, EPOCHS + 1):
        #-------------------------EPOCH-------------------------
        print("EPOCH: " + str(a))
        counter = 0
        done = False
        idx = 0
        while not done:
            try:
                img_tf = np.array(imgs[idx:idx+BATCH_SIZE], np.dtype('float32')) # splicing by batch size
                ax_tf = np.array(axes[idx:idx+BATCH_SIZE], np.dtype('float32'))
 
                if(ax_tf.size == 0):
                    assert 5 == 6
 
                lws, _, summary = sess.run([loss, optimizer, merged_summary_op], feed_dict={input_tensor:img_tf, y_true:ax_tf, training:True})
                idx = idx + BATCH_SIZE
                print(str(counter) + " - LOSS: " + str(lws))
                summary_writer.add_summary(summary, total_counter)
                total_counter = total_counter + 1
                counter = counter + 1
            except:
                done = True
            print("", flush=True, end='')
        save_path = saver.save(sess, "model", global_step=a)