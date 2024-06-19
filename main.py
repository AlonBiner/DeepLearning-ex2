import torch
import torch.nn as nn
from data_loading import get_train_loader, get_test_loader
from models import Autoencoder
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
    return train_losses, test_losses


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


def whole_train():
    batch_size = 32
    epochs = 20

    train_loader = get_train_loader(batch_size=batch_size, shuffle=True)
    test_loader = get_test_loader(batch_size=batch_size, shuffle=False)

    autoencoder = Autoencoder().to(DEVICE)
    device = next(autoencoder.parameters()).device
    print("Model is on device:", device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    train_loss, test_loss = train_model(autoencoder, criterion, optimizer, train_loader, test_loader, num_epochs=epochs)

    # Model saving and plotting
    model_name = autoencoder.__class__.__name__
    current_time = datetime.now().strftime("%H:%M:%S_%d-%B-%Y")
    lr = optimizer.param_groups[0]['lr']
    name = f"{model_name}_{current_time}_lr_{lr}"
    torch.save(autoencoder, f"models/{name}")
    plot([train_loss, test_loss], f"plots/{name}")
    print('Finished Training, saved', model_name, 'in models/ folder and saved plots in plots/ folder')

    PLOT_SIZE = 10
    autoencoder.eval()
    images, _ = next(iter(test_loader))
    reconstructed = autoencoder(images[:PLOT_SIZE].to(DEVICE))
    np_images = images[:PLOT_SIZE].detach().cpu().numpy()
    np_reconstructed = reconstructed.detach().cpu().numpy()
    plot_images(np_images, np_reconstructed, plots_size=PLOT_SIZE)


if __name__ == "__main__":
    whole_train()
