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
            data.append(des)
            tags.append(tag)

        self.clf = model(features_extractor, data, tags)
        print('Classifier trained')

    def test(self, testing_set):
        accepted = 0
        total = 0
        for image, tag in testing_set:
            total += 1
            img = cv2.imread(image)
            answer = self.clf.classify(img)
            if answer[0] == tag:
                accepted += 1
            print(accepted / total, answer, tag, sep='\t')


if __name__ == '__main__':
    training, testing = caltech101.load()

    classifier = Classifier(features.SURF(), models.BagOfWords, training)
    classifier.test(testing)
