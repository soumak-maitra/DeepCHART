# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BASE_CHANNELS, LATENT_DIM, IN_CHANNELS, OUT_CHANNELS, N_SECTION, Z_SECTION

class ResidualBlock3D(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=[k//2 for k in kernel_size])
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=[k//2 for k in kernel_size])

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpBlock3D, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=(2, 2, 2), padding=[k//2 for k in kernel_size], output_padding=1
            ),
            nn.ReLU(),
        )
        self.conv = ResidualBlock3D(in_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class VAEUNet3D(nn.Module):
    def __init__(self):
        super(VAEUNet3D, self).__init__()

        self.base_channels = BASE_CHANNELS
        self.latent_dim = LATENT_DIM

        k7 = (7, 7, 21)
        k5 = (5, 5, 15)
        k3 = (3, 3, 9)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(IN_CHANNELS, BASE_CHANNELS, kernel_size=k7, padding=[k//2 for k in k7]),
            nn.ReLU(),
            ResidualBlock3D(BASE_CHANNELS, kernel_size=k7)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(BASE_CHANNELS, BASE_CHANNELS * 2, kernel_size=k5, stride=2, padding=[k//2 for k in k5]),
            nn.ReLU(),
            ResidualBlock3D(BASE_CHANNELS * 2, kernel_size=k5)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(BASE_CHANNELS * 2, BASE_CHANNELS * 4, kernel_size=k3, stride=2, padding=[k//2 for k in k3]),
            nn.ReLU(),
            ResidualBlock3D(BASE_CHANNELS * 4, kernel_size=k3)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(BASE_CHANNELS * 4, BASE_CHANNELS * 8, kernel_size=k3, stride=2, padding=[k//2 for k in k3]),
            nn.ReLU(),
            ResidualBlock3D(BASE_CHANNELS * 8, kernel_size=k3)
        )

        # Latent space
        self.flat_size = BASE_CHANNELS * 8 * (N_SECTION//8) * (N_SECTION//8) * (Z_SECTION//8)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flat_size, LATENT_DIM)
        self.fc_logvar = nn.Linear(self.flat_size, LATENT_DIM)
        self.fc_dec = nn.Linear(LATENT_DIM, self.flat_size)

        # Decoder
        self.dec3 = UpBlock3D(BASE_CHANNELS * 8, BASE_CHANNELS * 4, kernel_size=k3)
        self.dec2 = UpBlock3D(BASE_CHANNELS * 4, BASE_CHANNELS * 2, kernel_size=k5)
        self.dec1 = UpBlock3D(BASE_CHANNELS * 2, BASE_CHANNELS, kernel_size=k7)

        self.final = nn.Conv3d(BASE_CHANNELS, OUT_CHANNELS, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        flat = self.flatten(e4)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        d4 = self.fc_dec(z).view(-1, BASE_CHANNELS * 8, N_SECTION//8, N_SECTION//8, Z_SECTION//8)
        d4 = d4 + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1

        return self.final(d1), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
