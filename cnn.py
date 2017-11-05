import cv2
import numpy as np
import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class CNN:
    def __init__(self, data, tags, side=32):
        i = 0
        self.categories = {}
        self.tags = []
        for t in tags:
            if t in self.categories:
                continue
            self.categories[t] = i
            self.tags.append(t)
            i += 1
        tags = [self.categories[t] for t in tags]

        self.side = side

        # Make sure the data is normalized
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Create extra synthetic training data by flipping, rotating and blurring the images on our data set
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)
        img_aug.add_random_blur(sigma_max=3.)

        # Define our network architecture:
        # Input is a 32x32 image with 3 color channels (red, green and blue)
        network = input_data(shape=[None, side, side, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

        # Step 1: Convolution
        network = conv_2d(network, side, 3, activation='relu')

        # Step 2: Max pooling
        network = max_pool_2d(network, 2)

        # Steps 3-4: Convolution again (twice)
        for _ in range(2):
            network = conv_2d(network, 2 * side, 3, activation='relu')

        # Step 5: Max pooling again
        network = max_pool_2d(network, 2)

        # Step 6: Fully-connected 2*side node neural network
        network = fully_connected(network, 2 * side, activation='relu')

        # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
        network = dropout(network, 0.5)

        # Step 8: Fully-connected neural network with as many outputs as different categories in the training set
        network = fully_connected(network, i, activation='softmax')

        # Tell tflearn how we want to train the network
        network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

        # Wrap the network in a model object
        self.model = tflearn.DNN(network, tensorboard_verbose=0)

        # Let's train it!
        self._train(data, tags)

    def _train(self, data, tags):
        data, tags = shuffle(data, tags)
        tags = to_categorical(tags, len(self.categories))
        images = []
        for image_path in data:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (self.side, self.side), interpolation=cv2.INTER_AREA)
            images.append(np.asarray(img[:, :], dtype=np.float32))
        self.model.fit(
            images, tags, n_epoch=50, shuffle=True, show_metric=True, batch_size=96, run_id='images-classifier-cnn'
        )

    def classify(self, image):
        img = cv2.resize(image, (self.side, self.side), interpolation=cv2.INTER_AREA)
        img = np.asarray(img[:, :], dtype=np.float32)
        prediction = self.model.predict([img])[0]
        return self.tags[max(range(len(prediction)), key=lambda i: prediction[i])]
