import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import build_ae_encoder, build_ae_decoder, build_vae_encoder, build_vae_decoder

def test_ae_shapes():
    latent_dim = 32
    dummy_input = tf.random.normal(shape=(1, 64, 64, 1))
    
    encoder = build_ae_encoder(latent_dim=latent_dim)
    decoder = build_ae_decoder(latent_dim=latent_dim)
    
    latent_output = encoder(dummy_input)
    assert tuple(latent_output.shape) == (1, latent_dim), f"AE Encoder failed. Expected (1, 32), got {latent_output.shape}"
    
    reconstructed = decoder(latent_output)
    assert tuple(reconstructed.shape) == (1, 64, 64, 1), f"AE Decoder failed. Expected (1, 64, 64, 1), got {reconstructed.shape}"

def test_vae_shapes():
    latent_dim = 32
    dummy_input = tf.random.normal(shape=(1, 64, 64, 1))
    
    encoder = build_vae_encoder(latent_dim=latent_dim)
    decoder = build_vae_decoder(latent_dim=latent_dim)
    
    z_mean, z_log_var, z = encoder(dummy_input)
    assert tuple(z_mean.shape) == (1, latent_dim), "VAE z_mean shape is incorrect."
    assert tuple(z_log_var.shape) == (1, latent_dim), "VAE z_log_var shape is incorrect."
    assert tuple(z.shape) == (1, latent_dim), "VAE sampled z shape is incorrect."
    
    reconstructed = decoder(z)
    assert tuple(reconstructed.shape) == (1, 64, 64, 1), "VAE Decoder shape is incorrect."

if __name__ == "__main__":
    test_ae_shapes()
    test_vae_shapes()
    print("All model tests passed successfully!")