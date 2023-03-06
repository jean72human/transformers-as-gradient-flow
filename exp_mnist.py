import torch
import torchvision
import torchvision.transforms as transforms

import argparse

from tqdm import tqdm

from models.vit import ViT, SimpleViT, SimpleViTI, SimpleViTFT
from utils import train

import torch.optim as optim

from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()


parser.add_argument('-m', '--method', choices=['SimpleViT','SimpleViTI','SimpleViTFT','ViT'], required=True)
parser.add_argument('-d', '--device', required=True)

config = parser.parse_args()

augment = True
batch_size = 128
num_epochs = 30
size = 32
patch_size = 8
learning_rate = 1e-5
device = config.device
data_path = '/home/gbetondji/smoothtransformer/data/'


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training {config.method}")

model_bases = {
    'SimpleViT':SimpleViT,
    'SimpleViTI':SimpleViTI,
    'SimpleViTFT':SimpleViTFT,
    'ViT':ViT
}

model = model_bases[config.method](
    image_size = size,
    patch_size = patch_size,
    num_classes = 10,
    dim = 16,
    depth = 32,
    heads = 1,
    mlp_dim = 1024,
    dropout = 0.,
    emb_dropout = 0.
)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):  
    train_loss, train_acc = train(model,train_loader,optimizer,criterion,epoch,device)  

    # Model validation
    model.eval()
    val_loss = val_acc = 0.0
    for (img, labels) in val_loader:
        img, labels = img.to(device), labels.to(device)
        with torch.no_grad():
            predictions = model(img)
            loss = criterion(predictions, labels)
            correct = torch.argmax(predictions.data, 1) == labels
        val_loss += loss
        val_acc += correct.sum()
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    print()
    print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.6f}")
    print(f" Valid acc {val_acc:.3f}, Val loss {val_loss:.6f}")
    
print(f"Trained {config.method}")
