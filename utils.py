import pathlib
from functools import reduce

import cv2


def image_generator(path):
    for p in pathlib.Path(path).iterdir():
        for img in p.iterdir():
            img_path = '%s/%s' % (img.cwd(), img)
            img = cv2.imread(img_path)
            yield img, p.name


def get_categories(path):
    return [c for c in pathlib.Path(path).iterdir()]


def get_size(path):
    result = 0
    for p in pathlib.Path(path).iterdir():
        result += reduce(lambda a, _: 1 + a, p.iterdir(), 0)
    return result
