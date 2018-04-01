from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, Reshape
from keras.preprocessing.image import ImageDataGenerator

input_tensor = Input((150, 200, 3), name='input_tensor')
input_tensor = Reshape((224, 224, 3))(input_tensor)
resnet = ResNet50(False, 'imagenet', model, (224, 224))
layers = resnet.output
layers = Flatten()(model)
layers = Dense(2)(model)
output = layers

model = Model(inputs=input_tensor, outputs=output)

for layer in resnet.layers[:75]:
    layer.trainable = False
for layer in resnet.layers[75:]:
    layer.trainable = True

print("Model Defined")

data = np.load("data/final.npy")
imgs = []
axes = []
for d in data:
	imgs.append(d[0])
	axes.append(d[1])
del data

print("Data Loaded")

imgs = np.array(imgs, np.dtype('float32'))
axes = np.array(axes, np.dtype('float32'))

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

train_datagen.fit(imgs)
print("Datagen Fit")

from keras.callbacks import TensorBoard
tfBoard = TensorBoard("./logs")
model.fit_generator(train_datagen, 32, 25, verbose=3, callbacks=[tfBoard])