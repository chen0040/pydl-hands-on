import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

model_architecture = '../../../cache/cifar10_architecture.json'
model_weights = '../../../cache/cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

img_names = ['../../../cache/cat.jpg', '../../../cache/dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
predictions = model.predict_classes(imgs)
print(predictions)