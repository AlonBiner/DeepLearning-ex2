# data.py
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# Download the training and test datasets
TRAIN_SET = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
TEST_SET = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


def get_train_loader(batch_size=64, shuffle=True, samples_num=None):
    if samples_num:
        indices = torch.arange(100)
        train_loader_CLS = torch.utils.data.Subset(TRAIN_SET, indices)
        return torch.utils.data.DataLoader(train_loader_CLS, batch_size=batch_size, shuffle=True, num_workers=0)
    return data.DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=shuffle)


def get_test_loader(batch_size=64, shuffle=False):
    return data.DataLoader(TEST_SET, batch_size=batch_size, shuffle=shuffle)


# Display the size of the datasets (for verification purposes)
if __name__ == "__main__":
    print(f"Number of training samples: {len(TRAIN_SET)}")
    print(f"Number of testing samples: {len(TEST_SET)}")
    pass
