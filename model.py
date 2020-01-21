import tensorflow as tf

layers = tf.keras.layers


def resnet_block(x, filters):
    y = layers.Conv3D(filters, (3, 3, 3), activation="relu", padding="same")(x)
    return layers.add([x, y])


def build_generator(image_size):
    image = layers.Input(shape=(image_size, image_size, 3, 1))
    y = layers.Conv3D(16, (3, 3, 3), activation="relu", padding="same")(image)
    y = resnet_block(y, 16)
    y = layers.Conv3D(1, (1, 1, 1), activation="relu")(y)
    y = layers.Reshape((image_size, image_size, 3, 1))(y)
    return tf.keras.Model(inputs=image, outputs=y)


def dense_conv(x, nfilter):
    y = layers.Conv3D(nfilter, (3, 3, 3), padding="same")(x)
    y = layers.LeakyReLU()(y)
    return layers.concatenate([x, y], axis=-1)


def build_critic(image_size):
    image = layers.Input(shape=(image_size, image_size, 3, 1))
    y = layers.Conv3D(16, (3, 3, 3), activation="relu", padding="same")(image)
    y = layers.LeakyReLU()(y)
    y = dense_conv(y, 32)
    y = layers.Flatten()(y)
    y = layers.Dense(1)(y)
    return tf.keras.Model(inputs=image, outputs=y)
