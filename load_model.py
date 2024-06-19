import torch
from data_loading import get_test_loader
from plots import plot_images

if __name__ == '__main__':
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 20

    test_loader = get_test_loader(batch_size=BATCH_SIZE, shuffle=False)
    images, _ = next(iter(test_loader))

    model_path = "Autoencoder_14:21:27_19-June-2024_lr_0.001.png"
    model = torch.load(f"models/{model_path}", map_location=DEVICE)
    reconstructed = model(images.to(DEVICE))

    np_images = images.detach().cpu().numpy()
    np_reconstructed = reconstructed.detach().cpu().numpy()
    plot_images(np_images, np_reconstructed, plots_size=BATCH_SIZE)
