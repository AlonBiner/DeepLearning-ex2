import torch

from data import get_train_loader, get_test_loader
from torch import nn

BATCH_SIZE = 64
LATENT_DIM = 12


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  # 32, 28, 28
            nn.MaxPool2d(2),  # 32, 14, 14
            nn.Conv2d(32, 64, 3, 1, padding=1),  # 64, 14, 14
            nn.MaxPool2d(2),  # 64, 7, 7
            nn.Conv2d(64, 128, 3, 1, padding=1),  # 128, 7, 7
            nn.MaxPool2d(2),  # 128, 3, 3
            nn.Conv2d(128, LATENT_DIM, 3, 1, 0),  # LATENT_DIM, 1, 1
        )

    def forward(self, x):
        output = self.encoder(x)
        return output


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        def __init__(self, input_dim, output_dim):
            super(MLP, self).__init__()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

        def forward(self, x):
            return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 128, 3, 1),
            torch.nn.MaxUnpool2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            torch.nn.MaxUnpool2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 1, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1, 1, 1)  # Reshape to (batch_size, latent_dim, 1, 1)
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.mlp = MLP(latent_dim, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.mlp(encoded)
        decoded = self.decoder(latent)
        return decoded


if __name__ == "__main__":
    train_loader = get_train_loader(batch_size=BATCH_SIZE, shuffle=True)
    test_loader = get_test_loader(batch_size=BATCH_SIZE, shuffle=False)
    sample_input = next(iter(train_loader))[0][0]  # Get the first image from the first batch
    print(sample_input.shape)

    encoder, decoder = Encoder(), Decoder()
    encoded_output = encoder(sample_input)
    print(encoded_output.shape)
    decoded_output = decoder(encoded_output)
    print(decoded_output.shape)
