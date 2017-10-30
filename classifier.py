import time

import cv2

import features
import models
from testing import caltech101


class Classifier:
    def __init__(self, features_extractor, model, training_set):
        data, tags = [], []

        for image, tag in training_set:
            img = cv2.imread(image)

            des = features_extractor.get_descriptors(img)
            if des is None:
                continue
            data.append(des)
            tags.append(tag)

        print('Starting training...')
        start = time.time()
        self.clf = model(features_extractor, data, tags)
        print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))

    def test(self, testing_set):
        accepted = 0
        total = 0
        for image, tag in testing_set:
            total += 1
            img = cv2.imread(image)
            answer = self.clf.classify(img)
            if answer == tag:
                accepted += 1
            print(accepted / total, answer, tag, sep='\t')


if __name__ == '__main__':
    training, testing = caltech101.load(15, 50)

    classifier = Classifier(features.SURF(), models.BagOfWords, training)
    classifier.test(testing)
