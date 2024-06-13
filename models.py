import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 13


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 28 -> 28
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 14 -> 14
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14 -> 7
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)  # 7 -> 5
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         return x

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


# Fully Connected Layers
# class FullyConnected(nn.Module):
#     def __init__(self):
#         super(FullyConnected, self).__init__()
#         self.fc1 = nn.Linear(in_features=128 * 5 * 5, out_features=LATENT_DIM)
#         self.fc2 = nn.Linear(in_features=LATENT_DIM, out_features=128 * 5 * 5)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = x.view(x.size(0), 128, 5, 5)
#         return x

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


# Decoder
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=0)  # 5 -> 7
#         self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
#                                           output_padding=1)  # 7 -> 14
#         self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1,
#                                           output_padding=1)  # 14 -> 28
#
#     def forward(self, x):
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = torch.tanh(self.deconv3(x))
#         return x

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


class GitAutoencoder(nn.Module):
    def __init__(self):
        super(GitAutoencoder, self).__init__()
        self.encoder = GitEncoder()
        self.decoder = GitDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class GitAutoencoder(nn.Module):
#     def __init__(self):
#         super(GitAutoencoder, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=5),
#             nn.ReLU(True),
#             nn.Conv2d(6, 16, kernel_size=5),
#             nn.ReLU(True))
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 6, kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(6, 1, kernel_size=5),
#             nn.ReLU(True),
#             nn.Tanh())
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
