import tensorflow as tf

layers = tf.keras.layers


def build_generator():
    image = layers.Input(shape=(100, 100, 3, 1))
    y = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(image)
    y = layers.Conv3D(1, (1, 1, 1), activation="relu")(y)
    y = layers.Reshape((100, 100, 3, 1))(y)
    return tf.keras.Model(inputs=image, outputs=y)


def build_critic():
    image = layers.Input(shape=(100, 100, 3, 1))
    y = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(image)
    y = layers.LeakyReLU()(y)
    y = layers.Flatten()(y)
    y = layers.Dense(1)(y)
    return tf.keras.Model(inputs=image, outputs=y)
