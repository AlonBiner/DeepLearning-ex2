import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 12


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28 -> 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14 -> 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # 7 -> 5
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, 128 * 5 * 5),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 5, 5)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3),  # 5 -> 7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.fc = FullyConnected()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = self.decoder(x)
        return x


# Working dimensions model. but not learning well at all.
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),  # 1x28x28 -> 16x24x24
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x24x24 -> 16x12x12
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),  # 16x12x12 -> 32x8x8
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x8x8 -> 32x4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # 32x4x4 -> 64x2x2
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 12),
            nn.ReLU(True),
            nn.Linear(12, 12),
            nn.ReLU(True),
            nn.Linear(12, 64 * 2 * 2),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0),  # 64x2x2 -> 32x5x5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),  # 32x5x5 -> 16x12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),  # 16x12x12 -> 16x24x24
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=0),  # 16x24x24 -> 1x28x28
            nn.Sigmoid()  # Optional: Use if you want the output to be normalized (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 64, 2, 2)
        x = self.decoder(x)
        return x


class GitAutoencoder(nn.Module):
    def __init__(self):
        super(GitAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))

        self.fc = nn.Sequential(
            nn.Linear(16 * 20 * 20, LATENT_DIM),  # Adjusted input size
            nn.ReLU(True),
            # nn.Linear(150, LATENT_DIM),
            # nn.ReLU(True),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.ReLU(True),
            # nn.Linear(LATENT_DIM, 150),
            # nn.ReLU(True),
            nn.Linear(LATENT_DIM, 16 * 20 * 20),  # Adjusted output size
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=5),
            nn.ReLU(True),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 16, 20, 20)
        x = self.decoder(x)
        return x
