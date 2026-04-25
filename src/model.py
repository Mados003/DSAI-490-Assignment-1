import tensorflow as tf


def build_ae_encoder(input_shape=(64, 64, 1), latent_dim=32):
    inputs = tf.keras.Input(shape=input_shape, name='ae_input')
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    latent_output = tf.keras.layers.Dense(latent_dim, name='ae_latent')(x)
    return tf.keras.Model(inputs, latent_output, name='ae_encoder')


def build_ae_decoder(latent_dim=32):
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='ae_latent_input')
    x = tf.keras.layers.Dense(16 * 16 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((16, 16, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return tf.keras.Model(latent_inputs, outputs, name='ae_decoder')


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_encoder(input_shape=(64, 64, 1), latent_dim=32):
    inputs = tf.keras.Input(shape=input_shape, name='vae_input')
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='vae_encoder')


def build_vae_decoder(latent_dim=32):
    return build_ae_decoder(latent_dim)