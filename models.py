import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 12


# Encoder
# 1x28x28 -> 16x24x24
# 16x24x24 -> 16x12x12 by maxpool
# 16x12x12 -> 32x8x8
# 32x8x8 -> 32x4x4 by maxpool
# 32x4x4 -> 64x2x2

# Fully connected:
# 64x2x2 -> 12 by fully connected
# 12 -> 64x2x2 by fully connected

# Decoder
# 64x2x2 -> 32x5x5 by convtranspose
# 32x5x5 -> 16x12x12 by convtranspose
# 16x12x12 -> 16x24x24 by convtranspose
# 16x24x24 -> 1x28x28 by convtranspose

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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

    def forward(self, x):
        return self.encoder(x)


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64 * 2 * 2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0),  # 64x2x2 -> 32x5x5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),  # 32x5x5 -> 16x12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),  # 16x12x12 -> 16x24x24
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=0),  # 16x24x24 -> 1x28x28
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 64, 2, 2)
        x = self.decoder(x)
        return x
