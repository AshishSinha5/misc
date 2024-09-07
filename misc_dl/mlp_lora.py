import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

# Load MNIST dataset with flattening transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

class LoraLayer(nn.Module):
    def __init__(self, inFeat, outFeat, rank=4):
        super(LoraLayer, self).__init__()
        self.inFeat = inFeat
        self.outFeat = outFeat
        self.rank = rank

        # initilaize the frozen LORA layer
        self.W = nn.Parameter(torch.Tensor(outFeat, inFeat), requires_grad=False)
        nn.init.uniform_(self.W, a=0.0, b=1.0)

        # initialize the LORA matrices 
        self.A = nn.Parameter(torch.Tensor(rank, inFeat))
        self.B = nn.Parameter(torch.Tensor(outFeat, rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.uniform_(self.A, a=0.0, b=1.0)
        nn.init.zeros_(self.B)

    def forward(self, x):
        return F.linear(x, self.W + torch.matmul(self.B, self.A), None)
        
class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc1 = LoraLayer(784, 128)
        self.fc2 = LoraLayer(128, 10)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x)
    

model = MNIST()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

# training loop 
def train(model, train_loader, optimizer, epochs):
    model.train()

    history = {
        "train_loss" : [],
        "train_acc" : []
    }
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == target).float().mean()
            history['train_loss'].append(loss.item())
            history['train_acc'].append(accuracy.item())
            if batch_idx % 100 == 0:
                print(f"Epoch = {epoch} Batch = {batch_idx} Train Loss = {loss.item()} Train Accuracy = {accuracy.item()}")

    return history


def test(model, test_loader):
    model.eval()

    history = {
        "test_loss": [],
        "test_acc": []
    }

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = F.nll_loss(output, target)
            accuracy = (output.argmax(dim=1) == target).float().mean()
            history["test_loss"].append(loss.item())
            history["test_acc"].append(accuracy.item())
            print(f"Test Loss = {loss.item()} Test Accuracy = {accuracy.item()}")


    return history


history = train(model, train_loader, optimizer, epochs=5)
test_history = test(model, test_loader)

import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(test_history["test_loss"], label="Test Loss")
plt.plot(test_history["test_acc"], label="Test Accuracy")

plt.legend()
plt.show()

                
