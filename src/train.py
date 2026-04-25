import json
import os
import sys

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_dataset
from src.model import build_ae_encoder, build_ae_decoder, build_vae_encoder, build_vae_decoder

BATCH_SIZE = 64
LATENT_DIM = 32
EPOCHS = 10
DATA_DIR = 'data/raw/Medical MNIST'
MODEL_SAVE_DIR = 'models/'

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU detected by TensorFlow. Training will run on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")

def main():
    configure_gpu()

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        return

    train_dataset, class_names = load_dataset(DATA_DIR, batch_size=BATCH_SIZE)

    ae_encoder = build_ae_encoder(latent_dim=LATENT_DIM)
    ae_decoder = build_ae_decoder(latent_dim=LATENT_DIM)
    vae_encoder = build_vae_encoder(latent_dim=LATENT_DIM)
    vae_decoder = build_vae_decoder(latent_dim=LATENT_DIM)

    ae_optimizer = tf.keras.optimizers.Adam(1e-3)
    vae_optimizer = tf.keras.optimizers.Adam(1e-3)

    ae_loss_history = []
    vae_recon_loss_history = []
    vae_kl_loss_history = []

    def train_step_ae(images):
        with tf.GradientTape() as tape:
            latent = ae_encoder(images, training=True)
            reconstructed = ae_decoder(latent, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(images, reconstructed))
        gradients = tape.gradient(loss, ae_encoder.trainable_variables + ae_decoder.trainable_variables)
        ae_optimizer.apply_gradients(zip(gradients, ae_encoder.trainable_variables + ae_decoder.trainable_variables))
        return loss

    def train_step_vae(images):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = vae_encoder(images, training=True)
            reconstructed = vae_decoder(z, training=True)

            recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(images, reconstructed))
            recon_loss *= (64 * 64)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + kl_loss

        trainable_vars = vae_encoder.trainable_variables + vae_decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        vae_optimizer.apply_gradients(zip(gradients, trainable_vars))
        return recon_loss, kl_loss

    print(f"Training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        ae_epoch_loss = tf.keras.metrics.Mean()
        vae_epoch_recon = tf.keras.metrics.Mean()
        vae_epoch_kl = tf.keras.metrics.Mean()
        
        for images, _ in train_dataset:
            ae_loss = train_step_ae(images)
            ae_epoch_loss(ae_loss)
            
            recon_loss, kl_loss = train_step_vae(images)
            vae_epoch_recon(recon_loss)
            vae_epoch_kl(kl_loss)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"AE Loss: {ae_epoch_loss.result():.4f} | "
              f"VAE Recon: {vae_epoch_recon.result():.4f} | "
              f"VAE KL: {vae_epoch_kl.result():.4f}")

        ae_loss_history.append(float(ae_epoch_loss.result().numpy()))
        vae_recon_loss_history.append(float(vae_epoch_recon.result().numpy()))
        vae_kl_loss_history.append(float(vae_epoch_kl.result().numpy()))

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    ae_encoder.save_weights(os.path.join(MODEL_SAVE_DIR, 'ae_encoder.weights.h5'))
    ae_decoder.save_weights(os.path.join(MODEL_SAVE_DIR, 'ae_decoder.weights.h5'))
    vae_encoder.save_weights(os.path.join(MODEL_SAVE_DIR, 'vae_encoder.weights.h5'))
    vae_decoder.save_weights(os.path.join(MODEL_SAVE_DIR, 'vae_decoder.weights.h5'))

    history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as history_file:
        json.dump(
            {
                'ae_loss_history': ae_loss_history,
                'vae_recon_loss_history': vae_recon_loss_history,
                'vae_kl_loss_history': vae_kl_loss_history,
            },
            history_file,
            indent=2,
        )
    print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    main()