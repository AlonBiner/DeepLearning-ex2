import torch
import torch.nn as nn
from data_loading import get_train_loader, get_test_loader
from models import Autoencoder, GitAutoencoder
from datetime import datetime
from plots import plot


# Training and Evaluation
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=20):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        # train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train_loss += loss.item() * images.size(0)

        # train_loss /= len(train_loader.dataset)
        train_loss = evaluate(model, train_loader, criterion)
        train_losses.append(train_loss)

        test_loss = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss}')

    model_name = model.__class__.__name__
    current_time = datetime.now().strftime("%H:%M:%S_%d-%B-%Y")
    torch.save(ae, f"models/{model_name}_{current_time}")
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


if __name__ == "__main__":
    BATCH_SIZE = 256

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    train_loader = get_train_loader(batch_size=BATCH_SIZE, shuffle=True)
    test_loader = get_test_loader(batch_size=BATCH_SIZE, shuffle=False)

    ae = Autoencoder().to(DEVICE)
    # ae = GitAutoencoder().to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-5)
    train_model(ae, criterion, optimizer, train_loader, test_loader, num_epochs=5)
