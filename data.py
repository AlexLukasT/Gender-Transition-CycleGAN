import tensorflow as tf
import numpy as np
import os
from PIL import Image


def get_generator(image_size, savedir, shuffle, male):
    def generator():
        filenames = sorted(os.listdir(savedir))
        gender = 0 if male else 1
        indices = np.array([int(f.split("_")[1]) == gender for f in filenames])
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                img = Image.open(os.path.join(savedir, filenames[i]))
                img.thumbnail((image_size, image_size))
                array = np.asarray(img) / 256
                yield array.reshape((image_size, image_size, 3, 1))

    return generator


def get_dataset(image_size, savedir="data", shuffle=False, male=True):
    kwargs = {
        "output_shapes": tf.TensorShape((image_size, image_size, 3, 1)),
        "output_types": tf.float32,
    }
    generator = get_generator(image_size, savedir, shuffle, male)
    dataset = tf.data.Dataset.from_generator(generator, **kwargs)
    return dataset
