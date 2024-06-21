import torch
import torch.nn as nn
from data_loading import get_train_loader, get_test_loader
from plots import save_model, plot_graph

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=20):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f' Test Loss: {test_loss:.4f},'
            f'Train Accuracy: {train_accuracy:.4f},'
            f' Test Accuracy: {test_accuracy:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies


def evaluate(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = correct / total

    return test_loss, accuracy


def classifier_train(model, device, batch_size=64, epochs=6, lr=1e-3, samples_num=None):

    train_loader = get_train_loader(batch_size=batch_size, shuffle=True, samples_num=samples_num)
    test_loader = get_test_loader(batch_size=batch_size, shuffle=False)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    res = train_model(model, criterion, optimizer, train_loader, test_loader, epochs)
    train_loss, test_loss, train_accuracy, test_accuracy = res

    description = f"_samples_{samples_num}_" if samples_num else ""
    save_model(model, optimizer, epochs, batch_size, description=description)
    plot_graph(model, train_loss, test_loss, train_accuracy, test_accuracy)


if __name__ == "__main__":
    #    classifier()
    pass
