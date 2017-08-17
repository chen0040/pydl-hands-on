from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import scipy.misc

def VGG_16(weight_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 244, 244)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 244, 244)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 244, 244)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 244, 244)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weight_path:
        model.load_weights(weight_path)

    return model

if __name__ == "__main__":
    im = scipy.misc.imresize(scipy.misc.imread('../../../data/cat.jpg'), (224, 224)).astype('float32')
    im = np.transpose(im, (2,0,1))
    im = np.expand_dims(im, axis=0)
    K.set_image_dim_ordering("th")

    # Test pretrained model
    model = VGG_16('../../../cache/vgg16_weights.h5')
    optimizer = SGD()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    out = model.predict(im)
    print(np.argmax(out))



