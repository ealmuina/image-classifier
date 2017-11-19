import os

import cv2
import keras
import keras.applications
from keras import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class BaseCNN:
    def __init__(self, model, categories, side=224):
        self.side = side
        self.model = model
        self.categories = categories

    def train(self, dataset, epochs=50):
        train_path = os.path.join(dataset, 'train')
        validation_path = os.path.join(dataset, 'validation')

        train_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: preprocess_input(x, mode='tf'),
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: preprocess_input(x, mode='tf')
        )

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.side, self.side),
            batch_size=32,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_path,
            target_size=(self.side, self.side),
            batch_size=32,
            class_mode='categorical')

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=np.math.ceil(train_generator.num_classes / 32),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=np.math.ceil(validation_generator.num_classes / 32))

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


class _FineTunedCNN(BaseCNN):
    def __init__(self, model_cls, categories, freezed_layers, side=224):
        self.freezed_layers = freezed_layers

        # create the base pre-trained model
        base_model = model_cls(weights='imagenet', include_top=False, input_shape=(side, side, 3))

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(len(categories), activation='softmax')(x)

        # this is the model we will training
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: training only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        super().__init__(model, categories, side)

    def train(self, dataset, epochs=25):
        # training the model on the new data for a few epochs
        super().train(dataset, 10)

        # freeze the first self._freezed_layers layers and unfreeze the rest:
        for layer in self.model.layers[:self.freezed_layers]:
            layer.trainable = False
        for layer in self.model.layers[self.freezed_layers:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
        super().train(dataset, epochs)


class InceptionV3(_FineTunedCNN):
    def __init__(self, categories, side=224):
        super(InceptionV3, self).__init__(keras.applications.InceptionV3, categories, 249, side)


class MobileNet(_FineTunedCNN):
    def __init__(self, categories, side=224):
        super(MobileNet, self).__init__(keras.applications.MobileNet, categories, 249, side)


class Xception(_FineTunedCNN):
    def __init__(self, categories, side=224):
        super(Xception, self).__init__(keras.applications.Xception, categories, 249, side)
