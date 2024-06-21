import torch
import torch.nn as nn
from data_loading import get_train_loader, get_test_loader
from plots import plot_graph, save_model
import copy

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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

        train_loss = evaulate(model, train_loader, criterion)
        train_losses.append(train_loss)

        test_loss = evaulate(model, test_loader, criterion)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss}')

    return train_losses, test_losses


def evaulate(model, dataloader, criterion):
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


def autoencoder_train(model, device, batch_size=64, epochs=17, lr=1e-3, save_encoder=False):
    train_loader = get_train_loader(batch_size=batch_size, shuffle=True)
    test_loader = get_test_loader(batch_size=batch_size, shuffle=False)

    model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss, test_loss = train_model(model, criterion, optimizer, train_loader, test_loader, epochs)

    description = "_Of_Question_1_" if save_encoder else ""
    save_model(model, optimizer, epochs, batch_size, description=description)
    if save_encoder:
        save_model(model.encoder, optimizer, epochs, batch_size, description=description)
    plot_graph(model, train_loss, test_loss)


if __name__ == "__main__":
    # autoencoder()
    # autoencoder_pretrained()
    pass
