import argparse
import importlib.util
import json
import time

import cv2
import keras.applications

import features
import models
from cnn import CNN, InceptionV3, BaseCNN, MobileNet, Xception


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
    def __init__(self, model, trained, training_set):
        data, tags = [], []

        for image, tag in training_set:
            data.append(image)
            tags.append(tag)

        if not trained:
            print('Starting training...')
            start = time.time()
            cnn = model(list(set(tags)))
            cnn.train(data, tags)
            print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))
        else:
            cnn = BaseCNN(model, load_imagenet_categories())

        super().__init__(cnn)


def test(dataset, classifier):
    training, testing = dataset.load()

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
        classifier = ClassicClassifier(features_extractor(), model, training)
        classifier.test(testing)

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
        classifier = CNNClassifier(model, classifier['trained'], training)
        classifier.test(testing)

    else:
        raise SyntaxError("Invalid configuration file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to .json configuration file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
        for t in config:
            spec = importlib.util.spec_from_file_location('dataset', t['dataset'] + '/__init__.py')
            dataset = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset)
            for classifier in t['classifiers']:
                test(dataset, classifier)


if __name__ == '__main__':
    main()
