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
    BATCH_SIZE = 128

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


    # e = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
    #                   nn.MaxPool2d(2, stride=2),  # 16x28x28 -> 16x14x14
    #                   nn.ReLU(True),
    #                   nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 32x14x14
    #                   nn.MaxPool2d(2, stride=2),  # 32x14x14 -> 32x7x7
    #                   nn.ReLU(True),
    #                   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32x7x7 -> 64x7x7
    #                   nn.MaxPool2d(2, stride=2),  # 64x7x7 -> 64x3x3
    #                   nn.ReLU(True),
    #                   )
    #
    # fc = nn.Sequential(
    #     nn.Linear(64 * 3 * 3, 12),
    #     nn.ReLU(),
    #     nn.Linear(12, 64 * 3 * 3),
    #     nn.ReLU()
    # )
    #
    # d = nn.Sequential(
    #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0),  # 64x3x3 -> 32x7x7
    #     nn.ReLU(True),
    #     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x7x7 -> 16x14x14
    #     nn.ReLU(True),
    #     nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x14x14 -> 1x28x28
    #     nn.ReLU(True),
    #     nn.Tanh()
    # )
    #
    # oe = e((torch.randn(1, 1, 28, 28)))
    # print(oe.shape)
    #
    # oe = oe.view(oe.size(0), -1)
    # ofc = fc(oe)
    # ofc = ofc.view(ofc.size(0), 64, 3, 3)
    # print(ofc.shape)
    #
    # od = d(ofc)
    # print(od.shape)
