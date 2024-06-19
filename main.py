import torch
import torch.nn as nn
from data_loading import get_train_loader, get_test_loader
from models import Autoencoder, GitAutoencoder, AE
from datetime import datetime
from plots import plot, plot_images

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Training and Evaluation
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=20):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        for images, _ in train_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, train_loader, criterion)
        train_losses.append(train_loss)

        test_loss = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss}')

    model_name = model.__class__.__name__
    current_time = datetime.now().strftime("%H:%M:%S_%d-%B-%Y")
    torch.save(model, f"models/{model_name}_{current_time}")
    plot([train_losses, test_losses], f"{model_name}_plot_{current_time}")
    print('Finished Training, saved', model_name, 'in models/ folder and saved plots in plots/ folder')


def evaluate(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item() * images.size(0)

    test_loss /= len(dataloader.dataset)
    return test_loss


def whole_train(model):
    model.to(DEVICE)
    BATCH_SIZE = 32

    train_loader = get_train_loader(batch_size=BATCH_SIZE, shuffle=True)
    test_loader = get_test_loader(batch_size=BATCH_SIZE, shuffle=False)

    print("Model output shape is:", model(torch.randn(1, 1, 28, 28).to(DEVICE)).shape)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=3)

    PLOT_SIZE = 5
    model.eval()
    images, _ = next(iter(test_loader))
    reconstructed = model(images[:PLOT_SIZE].to(DEVICE))
    np_images = images[:PLOT_SIZE].detach().cpu().numpy()
    np_reconstructed = reconstructed.detach().cpu().numpy()
    plot_images(np_images, np_reconstructed, plots_size=PLOT_SIZE)


if __name__ == "__main__":
    whole_train(AE())
    # whole_train(GitAutoencoder())
    # whole_train(GitAutoencoderMlp())
    #
    # i = torch.randn(1, 1, 28, 28)
    # encoder = nn.Sequential(
    #     nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),  # 1x28x28 -> 16x24x24
    #     nn.ReLU(True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16x24x24 -> 16x12x12
    #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),  # 16x12x12 -> 32x8x8
    #     nn.ReLU(True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32x8x8 -> 32x4x4
    #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # 32x4x4 -> 64x2x2
    # )
    #
    # mlp = nn.Sequential(
    #     nn.Linear(64 * 2 * 2, 12),
    #     nn.ReLU(True),
    #     nn.Linear(12, 12),
    #     nn.ReLU(True),
    #     nn.Linear(12, 64 * 2 * 2),
    #     nn.ReLU(True)
    # )
    #
    # decoder = nn.Sequential(
    #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0),  # 64x2x2 -> 32x4x4
    #     nn.ReLU(True),
    #     nn.Upsample(scale_factor=2, mode='nearest'),  # 32x4x4 -> 32x8x8
    #     nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=0),  # 32x8x8 -> 16x12x12
    #     nn.ReLU(True),
    #     nn.Upsample(scale_factor=2, mode='nearest'),  # 16x12x12 -> 16x24x24
    #     nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=0),  # 16x24x24 -> 1x28x28
    #     nn.Sigmoid()  # Optional: Use if you want the output to be normalized (0, 1)
    # )
    #
    # print(encoder(i).shape)
    # encoder_output = encoder(i)
    # mlp_output = mlp(encoder_output.view(i.size(0), -1))
    # decoder_output = decoder(mlp_output.view(i.size(0), 64, 2, 2))
    # print(decoder_output.shape)
