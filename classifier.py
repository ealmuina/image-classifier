import json
import time

import cv2
import keras.applications

import features
import models
from cnn import CNN, InceptionV3, BaseCNN, MobileNet
from testing import cifar10, cifar100, caltech101


def load_imagenet_categories(imagenet_class_index_path='imagenet_class_index.json'):
    with open(imagenet_class_index_path) as imagenet:
        imagenet = json.load(imagenet)
        return [imagenet[str(i)][1] for i in range(1000)]


class Classifier:
    def __init__(self, clf):
        self.clf = clf

    def test(self, testing_set):
        accepted = 0
        total = 0
        for img, tag in testing_set:
            total += 1
            if isinstance(img, str):
                img = cv2.imread(img)
            answer = self.clf.classify(img)
            if answer == tag:
                accepted += 1
            print(accepted / total, answer, tag, sep='\t')


class ClassicClassifier(Classifier):
    def __init__(self, features_extractor, model, training_set):
        data, tags = [], []

        for img, tag in training_set:
            if isinstance(img, str):
                img = cv2.imread(img)
            des = features_extractor.get_descriptors(img)
            if des is None:
                continue
            data.append(des)
            tags.append(tag)

        print('Starting training...')
        start = time.time()
        super().__init__(model(features_extractor, data, tags))
        print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))


class CNNClassifier(Classifier):
    def __init__(self, training_set):
        data, tags = [], []

        for image, tag in training_set:
            data.append(image)
            tags.append(tag)

        print('Starting training...')
        start = time.time()
        cnn = MobileNet(list(set(tags)))
        cnn.train(data, tags)
        # cnn = BaseCNN(keras.applications.Xception(), load_imagenet_categories())
        super().__init__(cnn)
        print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))


if __name__ == '__main__':
    training, testing = caltech101.load(categories=3, size_limit=50)

    # classifier = ClassicClassifier(features.SURF(), models.BagOfWords, training)
    classifier = CNNClassifier(training)
    classifier.test(testing)
