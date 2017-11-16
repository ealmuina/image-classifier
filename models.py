import random

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BagOfWords:
    def __init__(self, features_extractor, data, tags, n_clusters=None):
        if not n_clusters:
            n_clusters = len(set(tags)) * 5
        self.ftext = features_extractor
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.svm = SVC()
        self.scale = None
        self._train(data, tags)

    def _histogram(self, descriptors):
        tags = self.kmeans.predict(descriptors)
        histogram = [0] * self.kmeans.n_clusters
        for t in tags:
            histogram[t] += 1
        return histogram

    def _train(self, data, tags):
        # Train K-Means
        all_descriptors = []
        for img_descriptors in data:
            all_descriptors += [des for des in img_descriptors]
        self.kmeans.fit(all_descriptors)

        # Build histograms and train SVM
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
    def __init__(self, features_extractor, data, tags):
        self.tags = list(set(tags))

    def classify(self, image):
        return random.choice(self.tags)
