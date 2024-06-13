import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

import numpy as np


def plot(curves, file_name, step=5):
    xaxis_len_range = range(1, len(curves[0]) + 1)
    tickvals = list(range(1, len(curves[0]) + 1, step))
    if tickvals[-1] != len(curves[0]):
        tickvals.append(len(curves[0]))

    fig = px.line(x=xaxis_len_range, y=curves, title="Train & Test Losses")
    fig.update_layout(
        xaxis_title="Epoch number",
        yaxis_title="Loss",
        xaxis=dict(tickvals=tickvals, tickmode="array"),
    )

    for i, title in enumerate(["Train Loss", "Test Loss"]):
        fig.update_traces(name=title, selector=dict(name=f"wide_variable_{i}"))

    pio.write_image(fig, f"plots/{file_name}.png")
    fig.show()


def plot_images(original_images, reconstructed_images, plots_size=10):
    fig, axes = plt.subplots(2, plots_size, figsize=(plots_size * 2, 4))

    for i in range(plots_size):
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
