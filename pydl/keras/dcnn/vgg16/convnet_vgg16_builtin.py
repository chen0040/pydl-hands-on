from keras.models import Model
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy
import scipy.misc

K.set_image_dim_ordering('th')

model = VGG16(weights="imagenet", include_top=True)
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

im = scipy.misc.imresize(scipy.misc.imread('cat.jpg'), (224, 224)).astype('float32')
im = numpy.transpose(im, (2, 0, 1))

res = model.predict(im)
print(numpy.argmax(res))