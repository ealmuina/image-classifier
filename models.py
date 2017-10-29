from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BagOfWords:
    def __init__(self, features_extractor, data, tags, n_clusters=None):
        if not n_clusters:
            n_clusters = 5 * len(data)
        self.ftext = features_extractor
        self.kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
        self.svm = SVC()
        self.scale = None
        self.train(data, tags)

    def histogram(self, descriptors):
        tags = self.kmeans.predict(descriptors)
        histogram = [0] * self.kmeans.n_clusters
        for t in tags:
            histogram[t] += 1
        return histogram

    def classify(self, image):
        des = self.ftext.get_descriptors(image)
        histogram = [self.histogram(des)]
        histogram = self.scale.transform(histogram)
        return self.svm.predict(histogram)

    def train(self, data, tags):
        # Train K-Means
        all_descriptors = []
        for img_descriptors in data:
            all_descriptors += [des for des in img_descriptors]
        self.kmeans.fit(all_descriptors)

        # Build histograms and train SVM
        histogram = []
        for des in data:
            histogram.append(self.histogram(des))

        self.scale = StandardScaler().fit(histogram)
        histogram = self.scale.transform(histogram)

        self.svm.fit(histogram, tags)
