import cv2


class _FeaturesExtractor:
    def __init__(self, detector, computer):
        self.detector = detector
        self.computer = computer

    def get_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray, None)

    def get_descriptors(self, image):
        return self.get_features(image)[1]

    def get_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp = self.detector.detect(gray, None)
        return self.computer.compute(gray, kp)


class SIFT(_FeaturesExtractor):
    def __init__(self):
        sift = cv2.xfeatures2d.SIFT_create()
        super().__init__(sift, sift)


class SURF(_FeaturesExtractor):
    def __init__(self):
        surf = cv2.xfeatures2d.SURF_create(350)
        super().__init__(surf, surf)


class BRIEF(_FeaturesExtractor):
    def __init__(self):
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        super().__init__(star, brief)


class ORB(_FeaturesExtractor):
    def __init__(self):
        orb = cv2.ORB_create()
        super().__init__(orb, orb)


class FAST(_FeaturesExtractor):
    def __init__(self):
        surf = cv2.xfeatures2d.SURF_create()
        fast = cv2.FastFeatureDetector_create()
        super().__init__(fast, surf)
