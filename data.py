import tensorflow as tf
import numpy as np
import os
from PIL import Image


def get_generator(savedir, shuffle, male):
    def generator():
        filenames = sorted(os.listdir(savedir))
        gender = 0 if male else 1
        indices = np.array([int(f.split("_")[1]) == gender for f in filenames])
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                img = Image.open(os.path.join(savedir, filenames[i]))
                img.thumbnail((100, 100))
                array = np.asarray(img) / 256
                yield array.reshape((100, 100, 3, 1))
    return generator


def get_dataset(savedir="/home/alex/Downloads/crop_part1", shuffle=False, male=True):
    kwargs = {"output_shapes": tf.TensorShape((100, 100, 3, 1)),
              "output_types": tf.float32}
    generator = get_generator(savedir, shuffle, male)
    dataset = tf.data.Dataset.from_generator(generator, **kwargs)
    return dataset
