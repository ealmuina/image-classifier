import argparse
import json
import os
import time

import keras.applications

from cnn import CNN, InceptionV3, BaseCNN, MobileNet, Xception
import features
import models
import utils


def load_imagenet_categories(imagenet_class_index_path='imagenet_class_index.json'):
    with open(imagenet_class_index_path) as imagenet:
        imagenet = json.load(imagenet)
        return [imagenet[str(i)][1] for i in range(1000)]


class Classifier:
    def __init__(self, clf, dataset):
        self.clf = clf
        self.dataset = dataset

    def test(self, testing_path=None):
        if not testing_path:
            testing_path = os.path.join(self.dataset, 'test')
        testing_set = utils.image_generator(testing_path)
        accepted = 0
        total = 0
        for img, tag in testing_set:
            total += 1
            answer = self.clf.classify(img)
            if answer == tag:
                accepted += 1
            print(accepted / total, answer, tag, sep='\t')


class ClassicClassifier(Classifier):
    def __init__(self, features_extractor, model, dataset):
        print('Starting training...')
        start = time.time()
        super().__init__(model(features_extractor, dataset), dataset)
        print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))


class CNNClassifier(Classifier):
    def __init__(self, model, trained, dataset):
        if trained:
            cnn = BaseCNN(model, load_imagenet_categories())
        else:
            print('Starting training...')
            start = time.time()
            train_path = os.path.join(dataset, 'train')
            cnn = model(utils.get_categories(train_path))
            cnn.train(dataset)
            print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))

        super().__init__(cnn, dataset)


def test(dataset, classifier):
    if classifier['type'].lower() == 'classic':
        features_extractor = {
            'sift': features.SIFT,
            'surf': features.SURF,
            'fast': features.FAST,
            'brief': features.BRIEF,
            'orb': features.ORB
        }[classifier['features'].lower()]
        model = {
            'bagofwords': models.BagOfWords,
        }[''.join(classifier['model'].lower().split())]
        classifier = ClassicClassifier(features_extractor(), model, dataset)
        classifier.test()

    elif classifier['type'].lower() == 'cnn':
        input_shape = (224, 224, 3)
        if classifier['trained']:
            model = {
                'mobilenet': keras.applications.MobileNet(input_shape=input_shape),
                'inceptionresnetv2': keras.applications.InceptionResNetV2(input_shape=input_shape),
                'inceptionv3': keras.applications.InceptionV3(input_shape=input_shape),
                'xception': keras.applications.Xception(input_shape=input_shape)
            }[''.join(classifier['model'].lower().split())]
        else:
            model = {
                'cnn': CNN,
                'mobilenet': MobileNet,
                'inceptionv3': InceptionV3,
                'xception': Xception,
            }[''.join(classifier['model'].lower().split())]
        classifier = CNNClassifier(model, classifier['trained'], dataset)
        classifier.test()

    else:
        raise SyntaxError("Invalid configuration file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to .json configuration file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
        for t in config:
            for classifier in t['classifiers']:
                test(t['dataset'], classifier)


if __name__ == '__main__':
    main()
