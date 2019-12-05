import tensorflow as tf

layers = tf.keras.layers


def build_generator(image_size):
    image = layers.Input(shape=(image_size, image_size, 3, 1))
    y = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(image)
    y = layers.Conv3D(1, (1, 1, 1), activation="relu")(y)
    y = layers.Reshape((image_size, image_size, 3, 1))(y)
    return tf.keras.Model(inputs=image, outputs=y)


def build_critic(image_size):
    image = layers.Input(shape=(image_size, image_size, 3, 1))
    y = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(image)
    y = layers.LeakyReLU()(y)
    y = layers.Flatten()(y)
    y = layers.Dense(1)(y)
    return tf.keras.Model(inputs=image, outputs=y)
