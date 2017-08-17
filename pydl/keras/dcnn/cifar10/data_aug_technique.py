from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy  as np

NUM_TO_AUGMENT = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)

xtas, ytas = [], []

for i in range(X_train.shape[0]):
    x = X_train[i]
    x = x.reshape((1, ) + x.shape)
    num_aug = 0
    for x_aug in datagen.flow(x, batch_size=1, save_to_dir='/tmp', save_prefix='cifar10'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1

    # force stop after first iteration
    if i == 1:
        break
