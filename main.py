import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import get_train_loader, get_test_loader

BATCH_SIZE = 2
LATENT_DIM = 12


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)  # Output: 28x28
        self.pool1 = nn.MaxPool2d(2, return_indices=True)  # Output: 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)  # Output: 14x14
        self.pool2 = nn.MaxPool2d(2, return_indices=True)  # Output: 7x7
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)  # Output: 7x7
        self.pool3 = nn.MaxPool2d(2, return_indices=True)  # Output: 3x3
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)  # Output: 3x3
        self.conv5 = nn.Conv2d(256, LATENT_DIM, 3, 1, padding=0)  # Output: 1x1

    def forward(self, x):
        x = self.conv1(x)
        x, indices1 = self.pool1(x)
        size1 = x.size()
        x = self.conv2(x)
        x, indices2 = self.pool2(x)
        size2 = x.size()
        x = self.conv3(x)
        x, indices3 = self.pool3(x)
        size3 = x.size()
        x = self.conv4(x)
        x = self.conv5(x)
        return x, (indices1, indices2, indices3), (size1, size2, size3)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convT1 = nn.ConvTranspose2d(LATENT_DIM, 256, 3, 1)  # Output: 3x3
        self.convT2 = nn.ConvTranspose2d(256, 128, 3, 1, padding=1)  # Output: 3x3
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)  # Output: 7x7
        self.convT3 = nn.ConvTranspose2d(128, 64, 3, 1, padding=1)  # Output: 7x7
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)  # Output: 14x14
        self.convT4 = nn.ConvTranspose2d(64, 32, 3, 1, padding=1)  # Output: 14x14
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)  # Output: 28x28
        self.convT5 = nn.ConvTranspose2d(32, 1, 3, 1, padding=1)  # Output: 28x28
        self.tanh = nn.Tanh()

    def forward(self, x, indices, sizes):
        # x = x.view(x.size(0), LATENT_DIM, 1, 1)
        x = self.convT1(x)
        x = self.convT2(x)
        print("x shape", x.shape)
        print("size:", sizes[2])
        print("indices:", indices[2].shape)
        x = self.unpool1(x, indices=indices[2])
        # x = self.unpool1(x, indices=indices[2], output_size=sizes[2])
        # x = self.convT3(x)
        # x = self.unpool2(x, indices[1], output_size=sizes[1])
        # x = self.convT4(x)
        # x = self.unpool3(x, indices[0], output_size=sizes[0])
        # x = self.convT5(x)
        # x = self.tanh(x)
        return x


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
    sample_input = next(iter(train_loader))[0]  # Get the first image from the first batch
    print(sample_input.shape)

    encoder_ = Encoder()
    decoder_ = Decoder()
    encoded_output, inds, sizes = encoder_(sample_input)
    print(encoded_output.shape)

    decoded_output = decoder_(encoded_output, inds, sizes)
    print("Decoed output shape:", decoded_output.shape)
