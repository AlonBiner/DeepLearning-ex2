import numpy as np
import torch
from plots import reconstruct_images_plot
from plots import plot_images
import models
from autoencoder import autoencoder_train
from digit_classifier import classifier_train
from models import (Autoencoder, Encoder, Decoder,
                    DigitClassifier, MLP)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", DEVICE)


def load_model(model_path):
    return torch.load(model_path, map_location=DEVICE)


def Q1():
    ae = Autoencoder(Encoder(), Decoder())
    print("Q1 autoencoder get 28x28 and output:", ae(torch.randn((1, 1, 28, 28))).shape)
    autoencoder_train(model=ae, device=DEVICE, batch_size=64, epochs=1, lr=1e-3, save_encoder=True)


def Q2():
    dc = DigitClassifier(Encoder(), MLP())
    classifier_train(model=dc, device=DEVICE, batch_size=64, epochs=6, lr=1e-3, save_encoder=True)


def Q3():
    pretrained_encoder_path = \
        "archive/trained_data/models_data/Encoder_Of_Question_2_10-56-16__30-06-2024_lr_0.001_epochs_6_batch_64"
    pretrained_encoder = load_model(pretrained_encoder_path)
    pretrain_ae = Autoencoder(pretrained_encoder, Decoder(), train_encoder=False)
    autoencoder_train(model=pretrain_ae, device=DEVICE, batch_size=64, epochs=6, lr=1e-3, save_encoder=False)
    reconstruct_images_models(50)


def Q4():
    dc = DigitClassifier(Encoder(), MLP())
    classifier_train(model=dc, device=DEVICE, batch_size=32, epochs=50, lr=1e-3, samples_num=100)


def Q5():
    pre_encoder_path = "archive/trained_data/models_data/Encoder_Of_Question_1__15-32-31__28-06-2024_lr_0.001_epochs_25_batch_32"
    pre_encoder = load_model(pre_encoder_path)
    digit_classifier_pretrained = DigitClassifier(pre_encoder, MLP())
    classifier_train(digit_classifier_pretrained, device=DEVICE, batch_size=32, epochs=50, lr=1e-3, samples_num=100)


def reconstruct_images_models(image_num=20):
    """
    This function will reconstruct images using the models trained in Q1 and Q3
    It's will save the images in the folder "trained_data/reconstructed_images"
    The first image will be of Model Q1 and the second image will be of Model Q3
    :param image_num: number of images to reconstruct and compare of the two models
    :return:
    """
    Q1_model_path = \
        "archive/trained_data/models_data/Autoencoder_Of_Question_1__18-09-06__28-06-2024_lr_0.001_epochs_25_batch_64"

    reconstruct_images_plot(model_path=Q1_model_path, device=DEVICE, image_num=image_num)

    Q3_model_path = \
        "archive/trained_data/models_data/Autoencoder_11-41-42__30-06-2024_lr_0.001_epochs_6_batch_64"

    reconstruct_images_plot(model_path=Q3_model_path, device=DEVICE, image_num=image_num)


def reconstruct_images(image_num=10):
    """
    This function will reconstruct images of given one model only.
    :param image_num:
    :return:
    """
    model_path = \
        "archive/trained_data/models_data/Autoencoder_Of_Question_1__18-09-06__28-06-2024_lr_0.001_epochs_25_batch_64"

    reconstruct_images_plot(model_path=model_path, device=DEVICE, image_num=image_num)


if __name__ == '__main__':
    """
    Each Question in Practical part will be called by the Question number. 
    reconstruct_images_models() plotting the reconstructed images of the two models trained in Q1 and Q3
    But need to give it the models paths.
    By each function we can determine all we need to train the model.
    """
    models.LATENT_DIM = 12
    # Q1()
    # Q2()
    # Q3()
    # Q4()
    Q5()
    # reconstruct_images_models(image_num=70)
    # reconstruct_images(image_num=10)
    pass
