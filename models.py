import torch
import torch.nn as nn

LATENT_DIM = 12


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
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),  # 32x4x4 -> 32x2x2
            nn.ReLU(True)
        )
        self.linear = nn.Sequential(
            nn.Linear(32 * 2 * 2, LATENT_DIM),
            nn.ReLU(True))  # 128 -> 12

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Making the encoder output flat (into vector)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(LATENT_DIM, 32 * 2 * 2),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=0, output_padding=0),  # 32x2x2 -> 32x5x5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),  # 32x5x5 -> 16x12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),  # 16x12x12 -> 16x24x24
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=0),  # 16x24x24 -> 1x28x28
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 32, 2, 2)
        x = self.decoder(x)
        return x


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, _encoder, _decoder, train_encoder=True):
        super(Autoencoder, self).__init__()
        self.encoder = _encoder
        self.decoder = _decoder
        self.train_encoder = train_encoder

    def forward(self, x):
        if not self.train_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(LATENT_DIM, 50),
            nn.ReLU(True),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class DigitClassifier(nn.Module):
    def __init__(self, _encoder, _mlp):
        super(DigitClassifier, self).__init__()
        self.encoder = _encoder
        self.mlp = _mlp

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
