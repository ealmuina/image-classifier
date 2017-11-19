import os
import random

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import utils


class BagOfWords:
    def __init__(self, features_extractor, dataset, n_clusters=None):
        train_path = os.path.join(dataset, 'train')
        validation_path = os.path.join(dataset, 'validation')
        if not n_clusters:
            n_clusters = len(utils.get_categories(train_path)) * 5
        self.ftext = features_extractor
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.svm = SVC()
        self.scale = None
        self._train(train_path)
        self._score(validation_path)

    def _histogram(self, descriptors):
        tags = self.kmeans.predict(descriptors)
        histogram = [0] * self.kmeans.n_clusters
        for t in tags:
            histogram[t] += 1
        return histogram

    def _score(self, path):
        data = list(map(lambda x: self.ftext.get_descriptors(x[0]), utils.image_generator(path)))
        tags = list(map(lambda x: x[1], utils.image_generator(path)))

        histograms = [self._histogram(des) for des in data]
        histograms = self.scale.transform(histograms)
        print('Training score: %.2f' % self.svm.score(histograms, tags))

    def _train(self, path):
        data = list(map(lambda x: self.ftext.get_descriptors(x[0]), utils.image_generator(path)))
        tags = list(map(lambda x: x[1], utils.image_generator(path)))

        # Train K-Means
        all_descriptors = []
        for descriptors in data:
            if descriptors is not None:
                all_descriptors.extend(descriptors)
        self.kmeans.fit(all_descriptors)

        # Build histograms and training SVM
        histogram = []
        for des in data:
            histogram.append(self._histogram(des))

        self.scale = StandardScaler().fit(histogram)
        histogram = self.scale.transform(histogram)

        self.svm.fit(histogram, tags)

    def classify(self, image):
        des = self.ftext.get_descriptors(image)
        histogram = [self._histogram(des)]
        histogram = self.scale.transform(histogram)
        return self.svm.predict(histogram)[0]


class Random:
    def __init__(self, features_extractor, dataset):
        self.tags = utils.get_categories(os.path.join(dataset, 'train'))

    def classify(self, image):
        return random.choice(self.tags)
