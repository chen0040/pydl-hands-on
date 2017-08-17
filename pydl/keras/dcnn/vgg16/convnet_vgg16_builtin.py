from keras.models import Model
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np
import scipy.misc

np.random.seed(42)

model = VGG16(weights="imagenet", include_top=True)
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

im = scipy.misc.imresize(scipy.misc.imread('../../../data/cat.jpg'), (224, 224)).astype('float32')
im = np.transpose(im, (1, 0, 2))
im = np.expand_dims(im, axis=0)
K.set_image_dim_ordering("th")

res = model.predict(im)
print(np.argmax(res))