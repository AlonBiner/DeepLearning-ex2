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


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
                                     nn.MaxPool2d(2, stride=2),  # 16x28x28 -> 16x14x14
                                     nn.ReLU(True),
                                     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 32x14x14
                                     nn.MaxPool2d(2, stride=2),  # 32x14x14 -> 32x7x7
                                     nn.ReLU(True),
                                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32x7x7 -> 64x7x7
                                     nn.MaxPool2d(2, stride=2),  # 64x7x7 -> 64x3x3
                                     nn.ReLU(True),
                                     )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0),  # 64x3x3 -> 32x7x7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x7x7 -> 16x14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x14x14 -> 1x28x28
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class AE(nn.Module):
#     def __init__(self):
#         super(AE, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
#                           nn.MaxPool2d(2, stride=2),  # 16x28x28 -> 16x14x14
#                           nn.ReLU(True),
#                           nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 32x14x14
#                           nn.MaxPool2d(2, stride=2),  # 32x14x14 -> 32x7x7
#                           nn.ReLU(True),
#                           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32x7x7 -> 64x7x7
#                           nn.MaxPool2d(2, stride=2),  # 64x7x7 -> 64x3x3
#                           nn.ReLU(True),
#                           )
#
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 3 * 3, LATENT_DIM),
#             nn.ReLU(),
#             nn.Linear(LATENT_DIM, LATENT_DIM),
#             nn.Linear(LATENT_DIM, LATENT_DIM),
#             nn.Linear(LATENT_DIM, 64 * 3 * 3),
#             nn.ReLU()
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0),  # 64x3x3 -> 32x7x7
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x7x7 -> 16x14x14
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x14x14 -> 1x28x28
#             nn.ReLU(True),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#         # x = x.view(x.size(0), 64, 3, 3)
#         x = self.decoder(x)
#         return x
class GitEncoder(nn.Module):
    def __init__(self):
        super(GitEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        return x


class GitDecoder(nn.Module):
    def __init__(self):
        super(GitDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=5),
            nn.ReLU(True),
            nn.Tanh())

    def forward(self, x):
        x = self.decoder(x)
        return x


class GitAutoencoderByParts(nn.Module):
    def __init__(self):
        super(GitAutoencoderByParts, self).__init__()
        self.encoder = GitEncoder()
        self.decoder = GitDecoder()

    def forward(self, x):
        x = self.encoder(x)
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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=5),
            nn.ReLU(True),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
