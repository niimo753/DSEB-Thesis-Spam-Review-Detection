import cv2
import numpy as np

def load_data(path=r"att-data/s{}/{}.pgm", classes=40, images_per_class=10):
    data = []
    label = []
    for c in range(1, classes+1):
        for i in range(1, images_per_class+1):
            image = cv2.imread(path.format(c, i), 0)
            image = image.flatten()
            data.append(image)
            label.append(c)
    return np.array(data), np.array(label)