import argparse
import json
import os
import time

import keras.applications

import cnn
import features
import models
import segmentation
import utils


class Classifier:
    def __init__(self, clf, dataset):
        self.clf = clf
        self.dataset = dataset

    def test(self, confidence=0.5, testing_path=None):
        if not testing_path:
            testing_path = os.path.join(self.dataset, 'test')
        testing_set = utils.image_generator(testing_path)
        accepted = 0
        total = 0
        for img, tag in testing_set:
            total += 1
            answer_set = set()
            for frame in segmentation.sliding_frame(img):
                answer, prob = self.clf.classify(frame)
                if prob >= confidence:
                    answer_set.add(answer)
            if tag in answer_set:
                accepted += 1
            print(accepted / total, answer_set, tag, sep='\t')


class ClassicClassifier(Classifier):
    def __init__(self, features_extractor, model, dataset):
        print('Starting training...')
        start = time.time()
        super().__init__(model(features_extractor, dataset), dataset)
        print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))


class CNNClassifier(Classifier):
    def __init__(self, model, trained, dataset):
        if trained:
            nn = cnn.BaseCNN(model, utils.load_imagenet_categories())
        else:
            print('Starting training...')
            start = time.time()
            train_path = os.path.join(dataset, 'train')
            nn = model(utils.get_categories(train_path))
            nn.train(dataset)
            print('Classifier trained in %.2f minutes' % ((time.time() - start) / 60))

        super().__init__(nn, dataset)


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
                'simplecnn': cnn.SimpleCNN,
                'mobilenet': cnn.MobileNet,
                'inceptionv3': cnn.InceptionV3,
                'xception': cnn.Xception,
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
