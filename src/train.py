import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_dataset
from src.model import build_ae_encoder, build_ae_decoder, build_vae_encoder, build_vae_decoder

BATCH_SIZE = 64
LATENT_DIM = 32
EPOCHS = 10
DATA_DIR = 'data/raw/Medical MNIST'
MODEL_SAVE_DIR = 'models/'

def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    print("CUDA GPU not detected. Training will run on CPU.")
    return torch.device('cpu')

def main():
    device = get_device()

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        return

    train_dataset, class_names = load_dataset(DATA_DIR, batch_size=BATCH_SIZE)

    ae_encoder = build_ae_encoder(latent_dim=LATENT_DIM).to(device)
    ae_decoder = build_ae_decoder(latent_dim=LATENT_DIM).to(device)
    vae_encoder = build_vae_encoder(latent_dim=LATENT_DIM).to(device)
    vae_decoder = build_vae_decoder(latent_dim=LATENT_DIM).to(device)

    ae_optimizer = torch.optim.Adam(list(ae_encoder.parameters()) + list(ae_decoder.parameters()), lr=1e-3)
    vae_optimizer = torch.optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=1e-3)

    def train_step_ae(images):
        ae_optimizer.zero_grad()
        latent = ae_encoder(images)
        reconstructed = ae_decoder(latent)
        loss = F.mse_loss(reconstructed, images)
        loss.backward()
        ae_optimizer.step()
        return loss.item()

    def train_step_vae(images):
        vae_optimizer.zero_grad()
        z_mean, z_log_var, z = vae_encoder(images)
        reconstructed = vae_decoder(z)

        recon_loss = F.binary_cross_entropy(reconstructed, images, reduction='mean') * (64 * 64)
        kl_loss = -0.5 * torch.mean(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        total_loss = recon_loss + kl_loss

        total_loss.backward()
        vae_optimizer.step()
        return recon_loss.item(), kl_loss.item()

    print(f"Training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        ae_epoch_losses = []
        vae_epoch_recons = []
        vae_epoch_kls = []
        
        for images, _ in train_dataset:
            images = images.to(device)
            ae_loss = train_step_ae(images)
            ae_epoch_losses.append(ae_loss)
            
            recon_loss, kl_loss = train_step_vae(images)
            vae_epoch_recons.append(recon_loss)
            vae_epoch_kls.append(kl_loss)

        ae_epoch_loss = sum(ae_epoch_losses) / len(ae_epoch_losses)
        vae_epoch_recon = sum(vae_epoch_recons) / len(vae_epoch_recons)
        vae_epoch_kl = sum(vae_epoch_kls) / len(vae_epoch_kls)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"AE Loss: {ae_epoch_loss:.4f} | "
              f"VAE Recon: {vae_epoch_recon:.4f} | "
              f"VAE KL: {vae_epoch_kl:.4f}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(ae_encoder.state_dict(), os.path.join(MODEL_SAVE_DIR, 'ae_encoder.pth'))
    torch.save(ae_decoder.state_dict(), os.path.join(MODEL_SAVE_DIR, 'ae_decoder.pth'))
    torch.save(vae_encoder.state_dict(), os.path.join(MODEL_SAVE_DIR, 'vae_encoder.pth'))
    torch.save(vae_decoder.state_dict(), os.path.join(MODEL_SAVE_DIR, 'vae_decoder.pth'))

if __name__ == "__main__":
    main()