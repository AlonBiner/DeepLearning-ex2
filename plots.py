import os
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import datetime
from data_loading import get_test_loader


def get_time():
    return datetime.now().strftime("%H-%M-%S__%d-%m-%Y")


def get_unique_filename(filename):
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    return new_filename


def plot_images(original_images, reconstructed_images):
    fig, axes = plt.subplots(2, len(original_images), figsize=(len(original_images) * 2, 4))

    for i in range(len(original_images)):
        # Plot original images
        axes[0, i].imshow(np.squeeze(original_images[i]), cmap='gray')
        axes[0, i].axis('off')

        # Plot reconstructed images
        axes[1, i].imshow(np.squeeze(reconstructed_images[i]), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original Images', fontsize=14)
    axes[1, 0].set_title('Reconstructed Images', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_grid_images(original_images, reconstructed_images, name, images_per_row=10):
    # NOTE: USE THIS FUNC WITH EVEN NUMBER OF IMAGES. I.E len(original_images) divisible by images_per_row
    num_images = len(original_images)
    rows = num_images // images_per_row
    if num_images % images_per_row != 0:
        rows += 1  # add an extra row for remaining images

    fig, axes = plt.subplots(2 * rows, images_per_row, figsize=(images_per_row * 2, 4 * rows))

    for idx in range(num_images):
        row = idx // images_per_row
        col = idx % images_per_row

        # Plot original images
        axes[2 * row, col].imshow(np.squeeze(original_images[idx]), cmap='gray')
        axes[2 * row, col].axis('off')

        # Plot reconstructed images
        axes[2 * row + 1, col].imshow(np.squeeze(reconstructed_images[idx]), cmap='gray')
        axes[2 * row + 1, col].axis('off')

    for row in range(rows):
        axes[2 * row, 0].set_title('Original Images', fontsize=20)
        axes[2 * row + 1, 0].set_title('Reconstructed Images', fontsize=20)

    new_name = get_unique_filename(f"trained_data/reconstructed_images_plots/{name}.png")
    plt.tight_layout()
    plt.savefig(new_name)
    print(f"Reconstructed images plot saved at {new_name}")
    plt.show()


def reconstruct_images_plot(model_path, device, image_num=10, name="reconstructed_images"):
    test_loader = get_test_loader(batch_size=image_num, shuffle=False)

    images, _ = next(iter(test_loader))

    model = torch.load(model_path, map_location=device)

    reconstructed = model(images.to(device))
    np_images = images.detach().cpu().numpy()
    np_reconstructed = reconstructed.detach().cpu().numpy()
    plot_grid_images(np_images, np_reconstructed, name)


def plot(title, curves, path, titles, yaxis_title, step=5):
    xaxis_len_range = range(1, len(curves[0]) + 1)
    tickvals = list(range(1, len(curves[0]) + 1, step))
    if tickvals[-1] != len(curves[0]):
        tickvals.append(len(curves[0]))

    fig = px.line(x=xaxis_len_range, y=curves, title=title)
    fig.update_layout(
        xaxis_title="Epoch number",
        yaxis_title=yaxis_title,
        xaxis=dict(tickvals=tickvals, tickmode="array"),
    )

    for i, title in enumerate(titles):
        fig.update_traces(name=title, selector=dict(name=f"wide_variable_{i}"))

    pio.write_image(fig, f"{path}.png")
    fig.show()


def plot_graph(model, train_loss, test_loss, train_accuracy=None, test_accuracy=None):
    name = f"{model.__class__.__name__}_{get_time()}"
    loss_path = f"trained_data/models_plots/{name}_loss"
    plot("Train & Test Losses",
         [train_loss, test_loss],
         loss_path,
         ["Train Loss", "Test Loss"], yaxis_title="Loss")
    print("Loss plot saved at", f"{loss_path}.png")

    if train_accuracy:
        acc_path = f"trained_data/models_plots/{name}_accuracy"
        plot("Train & Test Accuracies",
             [train_accuracy, test_accuracy],
             acc_path,
             ["Train Accuracy", "Test Accuracy"], yaxis_title="Accuracy")
        print("Accuracy plot saved at", f"{acc_path}.png")


def save_model(model, optimizer, epochs, batch_size, description=""):
    model_name = model.__class__.__name__ + description
    lr = optimizer.param_groups[0]['lr']
    name = f"{model_name}_{get_time()}_lr_{lr}_epochs_{epochs}_batch_{batch_size}"
    path = f"trained_data/models_data/{name}"
    torch.save(model, path)
    print(f'Finished Training. {model_name} saved at {path}')
