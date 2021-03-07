import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class mnistDataset(Dataset):
    def __init__(self, file_path, transform=True):
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485], std=[0.229])
        self.data = pd.read_csv(file_path)
        self.imgs = np.asarray(self.data.iloc[:, 1:])
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        x = self.imgs[index]
        x = torch.tensor(x)
        x = x / 255
        y = self.labels[index]
        return x, y


if __name__ == "__main__":
    trainset = mnistDataset(file_path='data/train.csv')
    train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
    for images, labels in train_loader:
        print(images[0])
        print(labels)
        break
