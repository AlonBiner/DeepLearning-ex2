import torch
from data_loading import get_test_loader


def classifier(model, device):
    BATCH_SIZE = 10

    test_loader = get_test_loader(batch_size=BATCH_SIZE, shuffle=False)
    images, labels = next(iter(test_loader))

    model_path = ""
    model = torch.load(f"models/{model_path}", map_location=DEVICE)
    outputs = model(images.to(DEVICE))

    _, predicted = torch.max(outputs, 1)

    print("True Labels  :", end=" ")
    for label in labels:
        print(f"{label:2}", end=" ")

    print("\nPredicted    :", end=" ")
    for pred in predicted:
        print(f"{pred:2}", end=" ")

    print("\n")


if __name__ == '__main__':
    # classifier()
    pass
