import torch
import torch.nn as nn
import torch.nn.functional as F


class AEEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 64, 128)
        self.fc_latent = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc_latent(x)


class AEDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 16 * 16 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 64, 16, 16)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return torch.sigmoid(self.deconv3(x))


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 64, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

    @staticmethod
    def reparameterize(z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + epsilon * std


def build_ae_encoder(input_shape=(64, 64, 1), latent_dim=32):
    return AEEncoder(latent_dim=latent_dim)


def build_ae_decoder(latent_dim=32):
    return AEDecoder(latent_dim=latent_dim)


def build_vae_encoder(input_shape=(64, 64, 1), latent_dim=32):
    return VAEEncoder(latent_dim=latent_dim)


def build_vae_decoder(latent_dim=32):
    return AEDecoder(latent_dim=latent_dim)