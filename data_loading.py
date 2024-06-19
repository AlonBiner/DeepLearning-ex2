# data.py
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download the training and test datasets
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


def get_train_loader(batch_size=64, shuffle=True):
    return data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)


def get_test_loader(batch_size=64, shuffle=False):
    return data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)


# Display the size of the datasets (for verification purposes)
if __name__ == "__main__":
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of testing samples: {len(test_set)}")
