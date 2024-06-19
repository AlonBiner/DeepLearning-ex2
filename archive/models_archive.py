# Working dimensions model. but not learning well at all.
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
            nn.MaxPool2d(2, stride=2),  # 16x28x28 -> 16x14x14
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 32x14x14
            nn.MaxPool2d(2, stride=2),  # 32x14x14 -> 32x7x7
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32x7x7 -> 64x7x7
            nn.MaxPool2d(2, stride=2),  # 64x7x7 -> 64x3x3
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, LATENT_DIM),
            nn.ReLU(True),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.ReLU(True),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.ReLU(True),
            nn.Linear(LATENT_DIM, 64 * 3 * 3),
            nn.ReLU(True)
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
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = x.view(x.size(0), 64, 3, 3)
        x = self.decoder(x)
        return x


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 16x28x28 -> 16x14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 32x14x14
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 32x14x14 -> 32x7x7
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 300),
            nn.ReLU(True),
            nn.Linear(300, LATENT_DIM),
            nn.ReLU(True),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.ReLU(True),
            nn.Linear(LATENT_DIM, 300),
            nn.ReLU(True),
            nn.Linear(300, 32 * 7 * 7),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x7x7 -> 16x14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x14x14 -> 1x28x28
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = x.view(x.size(0), 32, 7, 7)
        x = self.decoder(x)
        return x


## The following seem to work a little bit fine
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