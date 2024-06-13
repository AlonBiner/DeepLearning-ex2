import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_train_loader, get_test_loader

BATCH_SIZE = 2
LATENT_DIM = 12


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)  # 28 -> 14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # 14 -> 7
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)  # 7 -> 5

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


# Fully Connected Layers
class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_features=128 * 5 * 5, out_features=LATENT_DIM)
        self.fc2 = nn.Linear(in_features=LATENT_DIM, out_features=128 * 5 * 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 5, 5)
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=0)  # 5 -> 7
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)  # 7 -> 14
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)  # 14 -> 28

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x


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


# Training and Evaluation
def train_model(model, criterion, optimizer, dataloader, num_epochs=20):
    for epoch in range(num_epochs):
        for images, _ in dataloader:
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluate(model, dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('Finished Training')


# Evaluation: Comparing input vs. reconstructed images
def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images)
            # Here you can visualize or save the input and output images for comparison
            break  # Just for the example, evaluate on one batch


if __name__ == "__main__":
    train_loader = get_train_loader(batch_size=BATCH_SIZE, shuffle=True)
    test_loader = get_test_loader(batch_size=BATCH_SIZE, shuffle=False)
    sample_input = next(iter(train_loader))[0]  # Get the first image from the first batch
    print(sample_input.shape)

    encoder_ = Encoder()
    fully_connected_ = FullyConnected()
    decoder_ = Decoder()
    ae = Autoencoder()
    decoded_output = ae(sample_input)

    print("Decoded output shape:", decoded_output.shape)
