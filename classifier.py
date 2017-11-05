import time

import cv2

from cnn import CNN
from testing import cifar10


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
        super().__init__(CNN(data, tags))
        print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))


if __name__ == '__main__':
    training, testing = cifar10.load()

    # classifier = ClassicClassifier(features.SURF(), models.BagOfWords, training)
    classifier = CNNClassifier(training)
    classifier.test(testing)
