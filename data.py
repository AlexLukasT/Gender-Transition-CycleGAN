import tensorflow as tf
import numpy as np
import os
from PIL import Image


def get_indices(filenames, male):
    gender = 0 if male else 1
    indices = []
    for i, filename in enumerate(filenames):
        if int(filename.split("_")[1]) == gender:
            indices.append(i)
    indices = np.array(indices)
    n_train = int(len(indices) * 0.9)
    return indices[:n_train], indices[n_train:]


def get_generator(image_size, savedir, shuffle, male, test):
    def generator():
        filenames = sorted(os.listdir(savedir))
        train_indices, test_indices = get_indices(filenames, male)
        indices = train_indices if not test else test_indices
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                img = Image.open(os.path.join(savedir, filenames[i]))
                img.thumbnail((image_size, image_size))
                array = np.asarray(img) / 256
                yield array.reshape((image_size, image_size, 3, 1))

    return generator


def get_dataset(
    image_size, savedir="data", shuffle=False, male=True, test=False
):
    kwargs = {
        "output_shapes": tf.TensorShape((image_size, image_size, 3, 1)),
        "output_types": tf.float32,
    }
    generator = get_generator(image_size, savedir, shuffle, male, test)
    dataset = tf.data.Dataset.from_generator(generator, **kwargs)
    return dataset
