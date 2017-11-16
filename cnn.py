import json

import cv2
import keras
from keras import applications
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tflearn
from tflearn import local_response_normalization
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge


class BaseCNN:
    def __init__(self, model, categories, side=224):
        self.side = side
        self.model = model
        self.categories = categories

    def _get_image_generator(self, data):
        generator = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        generator.fit(self._load_images(data))
        return generator

    def _load_images(self, data):
        images = []
        for img in data:
            if isinstance(img, str):
                img = cv2.imread(img)
            img = cv2.resize(img, (self.side, self.side), interpolation=cv2.INTER_AREA)
            images.append(np.asarray(img[:, :], dtype=np.float32))
        return np.array(images)

    @property
    def _tag_to_number(self):
        tag_to_number = {self.categories[i]: i for i in range(len(self.categories))}
        return tag_to_number

    def train(self, data, data_tags, validation=None, validation_tags=None):
        tag_to_number = self._tag_to_number
        data_tags = list(map(lambda tag: tag_to_number[tag], data_tags))
        data_tags = keras.utils.to_categorical(data_tags, len(self.categories))
        data = self._load_images(data)

        generator = self._get_image_generator(data)
        self.model.fit_generator(
            generator.flow(self._load_images(data), data_tags, shuffle=True),
            steps_per_epoch=np.math.ceil(len(data) / 32),
            epochs=100,
            validation_data=(
                validation,
                keras.utils.to_categorical(list(map(lambda tag: tag_to_number[tag], validation_tags)),
                                           len(self.categories))) if validation and validation_tags else None
        )

    def classify(self, img):
        img = cv2.resize(img, (self.side, self.side), interpolation=cv2.INTER_AREA)
        img = np.asarray(img[:, :], dtype=np.float32)
        img = preprocess_input(img, mode='tf')
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        return self.categories[np.argmax(prediction)]


class CNN(BaseCNN):
    def __init__(self, categories, side=32):
        model = Sequential([
            Convolution2D(32, 3, activation='relu', input_shape=(side, side, 3)),
            MaxPooling2D((2, 2)),
            Convolution2D(64, 3, activation='relu'),
            Convolution2D(64, 3, activation='relu'),
            MaxPooling2D((2, 2)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Flatten(),
            Dense(len(categories), activation='softmax')
        ])
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')
        super().__init__(model, categories, side)


class AlexNet(BaseCNN):
    def __init__(self, data, tags, side=32):
        super().__init__(tags, side)
        network = self.network

        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, len(self.int_to_tag), activation='softmax')

        # Tell tflearn how we want to train the network
        network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)

        # Wrap the network in a model object
        self.model = tflearn.DNN(network, tensorboard_verbose=2)

        # Let's train it!
        self._train(data, self.numeric_tags)


class GoogLeNet(BaseCNN):
    def __init__(self, data, tags, side=32):
        super().__init__(tags, side)
        network = self.network

        conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

        # 3a
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu',
                                   name='inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_3a_5_5_reduce')
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu',
                                   name='inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu',
                                        name='inception_3a_pool_1_1')
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                    mode='concat', axis=3)

        # 3b
        inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                          name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu',
                                   name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                          name='inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_3b_pool_1_1')
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                    mode='concat', axis=3, name='inception_3b_output')
        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

        # 4a
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu',
                                          name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu',
                                   name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu',
                                   name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4a_pool_1_1')
        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                    mode='concat', axis=3, name='inception_4a_output')

        # 4b
        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                          name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu',
                                   name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                          name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4b_5_5')
        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4b_pool_1_1')
        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                    mode='concat', axis=3, name='inception_4b_output')

        # 4c
        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                          name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu',
                                   name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                          name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4c_5_5')
        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4c_pool_1_1')
        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                    mode='concat', axis=3, name='inception_4c_output')

        # 4d
        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                          name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu',
                                   name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                          name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4d_pool_1_1')
        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                    mode='concat', axis=3, name='inception_4d_output')

        # 4e
        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                          name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                          name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                        name='inception_4e_pool_1_1')
        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],
                                    axis=3, mode='concat')
        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

        # 5a
        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu',
                                          name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu',
                                          name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5a_pool_1_1')
        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
                                    axis=3, mode='concat')

        # 5b
        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                          name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu',
                                   name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                          name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5b_5_5')
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],
                                    axis=3, mode='concat')
        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)

        # fc
        loss = fully_connected(pool5_7_7, len(self.int_to_tag), activation='softmax')
        network = regression(loss, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)

        # Wrap the network in a model object
        self.model = tflearn.DNN(network, tensorboard_verbose=2)

        # Let's train it!
        self._train(data, self.numeric_tags)


class NIN(BaseCNN):
    def __init__(self, data, tags, side=32):
        super().__init__(tags, side)
        network = self.network

        network = conv_2d(network, 192, 5, activation='relu')
        network = conv_2d(network, 160, 1, activation='relu')
        network = conv_2d(network, 96, 1, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = dropout(network, 0.5)
        network = conv_2d(network, 192, 5, activation='relu')
        network = conv_2d(network, 192, 1, activation='relu')
        network = conv_2d(network, 192, 1, activation='relu')
        network = avg_pool_2d(network, 3, strides=2)
        network = dropout(network, 0.5)
        network = conv_2d(network, 192, 3, activation='relu')
        network = conv_2d(network, 192, 1, activation='relu')
        network = conv_2d(network, len(self.int_to_tag), 1, activation='relu')
        network = avg_pool_2d(network, 8)
        network = flatten(network)

        # Tell tflearn how we want to train the network
        network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

        # Wrap the network in a model object
        self.model = tflearn.DNN(network, tensorboard_verbose=2)

        # Let's train it!
        self._train(data, self.numeric_tags)


class VGG16:
    def __init__(self, data, tags, side=224, weights_path=None):
        self.datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        # self.datagen.fit(data)

        i = 0
        self.tag_to_int = {}
        self.int_to_tag = []
        for t in tags:
            if t in self.tag_to_int:
                continue
            self.tag_to_int[t] = i
            self.int_to_tag.append(t)
            i += 1
        self.numeric_tags = [self.tag_to_int[t] for t in tags]
        self.side = side

        model = applications.MobileNet(weights='imagenet')
        model = applications.Xception(weights='imagenet', input_shape=(side, side, 3))
        # model = applications.InceptionV3(weights='imagenet')
        # model = applications.InceptionResNetV2(weights='imagenet')

        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy')

        self.model = model
        # self._train(data, self.numeric_tags)

    def _train(self, data, tags):
        tags = keras.utils.to_categorical(tags, 100)
        images = []
        for img in data:
            if isinstance(img, str):
                img = cv2.imread(img)
            img = cv2.resize(img, (self.side, self.side), interpolation=cv2.INTER_AREA)
            images.append(np.asarray(img[:, :], dtype=np.float32))
        images = np.array(images)
        self.model.fit_generator(
            self.datagen.flow(
                images, tags,
                batch_size=32
            ),
            steps_per_epoch=int(np.ceil(images.shape[0] / float(32))),
            epochs=1,
            workers=4
        )

    def classify(self, img):
        img = cv2.resize(img, (self.side, self.side), interpolation=cv2.INTER_AREA)
        img = np.asarray(img[:, :], dtype=np.float32)
        img = preprocess_input(img, mode='tf')
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        # return self.int_to_tag[np.argmax(prediction)]
        return decode_predictions(prediction, top=1)[0][0][1]
