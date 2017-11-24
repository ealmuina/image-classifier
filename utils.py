import pathlib
from functools import reduce

import cv2


def image_generator(path):
    for p in pathlib.Path(path).iterdir():
        for img in p.iterdir():
            img = cv2.imread(str(img))
            yield img, p.name


def get_categories(path):
    return [c.name for c in pathlib.Path(path).iterdir()]


def get_size(path):
    result = 0
    for p in pathlib.Path(path).iterdir():
        result += reduce(lambda a, _: 1 + a, p.iterdir(), 0)
    return result


def load_imagenet_categories(imagenet_class_index_path='imagenet_class_index.json'):
    with open(imagenet_class_index_path) as imagenet:
        imagenet = json.load(imagenet)
        return [imagenet[str(i)][1] for i in range(1000)]
