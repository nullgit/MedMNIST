import cv2
import numpy as np
from math import ceil
import random

def data_resize(images):
    # print(images.shape)
    ret = []
    for i in range(images.shape[0]):
        ret.append(cv2.resize(images[i], (224, 224)))
    ret = np.array(ret)

    size = ret.shape
    if (len(images.shape) == 4):
        ret = ret.reshape(size[0], size[1], size[2], images.shape[-1])
    else:
        ret = ret.reshape(size[0], size[1], size[2], 1)
    # print(ret.shape)
    return ret


def data_generate(images, labels, batch_size=1):
    size = images.shape
    max_num = ceil(size[0] / batch_size)
    i = 0
    while True:
        batch_images = data_resize(images[i * batch_size: (i + 1) * batch_size])
        batch_labels = labels[i * batch_size: (i + 1) * batch_size]
        index = [j for j in range(len(batch_images))] 
        random.shuffle(index)
        batch_images = batch_images[index]
        batch_labels = batch_labels[index]
        # yield ({"input_1": batch_images}, {"dense_2": batch_labels})
        yield ({"input_1": batch_images}, {"dense_1": batch_labels}) # tf版本为2.4.1时，使用这条代码
        i = (i + 1) % max_num

def test_data_generate(images, labels, batch_size=1):
    size = images.shape
    max_num = ceil(size[0] / batch_size)
    i = 0
    while True:
        batch_images = data_resize(images[i * batch_size: (i + 1) * batch_size])
        batch_labels = labels[i * batch_size: (i + 1) * batch_size]
        # yield ({"input_1": batch_images}, {"dense_2": batch_labels})
        yield ({"input_1": batch_images}, {"dense_1": batch_labels}) # tf版本为2.4.1时，使用这条代码
        i = (i + 1) % max_num
