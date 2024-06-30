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
    autoencoder_train(model=ae, device=DEVICE, batch_size=64, epochs=25, lr=1e-3, save_encoder=True)


def Q2():
    dc = DigitClassifier(Encoder(), MLP())
    classifier_train(model=dc, device=DEVICE, batch_size=64, epochs=6, lr=1e-3, save_encoder=True)


def Q3():
    pretrained_encoder_path = \
        "trained_data/models_data/DigitClassifierEncoder_of_question2_09-58-42__29-06-2024_lr_0.001_epochs_6_batch_64"
    pretrained_encoder = load_model(pretrained_encoder_path)

    # Print encoder output shape for verification
    dummy_input = torch.randn(1, 1, 28, 28).to(DEVICE)
    encoder_output = pretrained_encoder(dummy_input)
    print(f'Encoder output shape: {encoder_output.shape}')

    pretrain_ae = Autoencoder(pretrained_encoder, Decoder(), train_encoder=False)
    autoencoder_train(model=pretrain_ae, device=DEVICE, batch_size=64, epochs=6, lr=1e-3, save_encoder=False)


def Q4():
    dc = DigitClassifier(Encoder(), MLP())
    classifier_train(model=dc, device=DEVICE, batch_size=32, epochs=50, lr=1e-3, samples_num=100)


def Q5():
    pre_encoder_path = "trained_data/models_data/Encoder_Of_Question_1__15-32-31__28-06-2024_lr_0.001_epochs_25_batch_32"
    pre_encoder = load_model(pre_encoder_path)
    digit_classifier_pretrained = DigitClassifier(pre_encoder, MLP())
    classifier_train(digit_classifier_pretrained, device=DEVICE, batch_size=32, epochs=50, lr=1e-3, samples_num=100)
    # classifier_train(digit_classifier_pretrained, device=DEVICE, batch_size=32, epochs=25, lr=1e-3, samples_num=100)

def reconstruct_images(image_num=20):
    # Q1_model_path = \
    #     "trained_data/models_data/Autoencoder_Of_Question_1__15-19-30__21-06-2024_lr_0.001_epochs_8_batch_64"
    Q1_model_path = \
         "trained_data/models_data/Autoencoder_Of_Question_1__18-09-06__28-06-2024_lr_0.001_epochs_25_batch_64"

    reconstruct_images_plot(model_path=Q1_model_path, device=DEVICE, image_num=image_num)

    Q3_model_path = \
        "trained_data/models_data/Autoencoder_Of_Question_1__15-19-30__21-06-2024_lr_0.001_epochs_8_batch_64"
    reconstruct_images_plot(model_path=Q3_model_path, device=DEVICE, image_num=image_num)


if __name__ == '__main__':
    models.LATENT_DIM = 12

    # Q1()
    # Q2()
    Q3()
    # Q4()
    # Q5()
    # reconstruct_images(image_num=10)

    # path = \
    #      "trained_data/models_data/Autoencoder_Of_Question_1__16-55-32__28-06-2024_lr_0.001_epochs_25_batch_64"
    # reconstruct_images_plot(model_path=path, device=DEVICE, image_num=10)

    pass
