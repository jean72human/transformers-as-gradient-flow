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
size = 128
patch_size = 16
learning_rate = 1e-4
device = config.device
train_path = '/home/gbetondji/smoothtransformer/data/imagewoof2-320/train'
val_path = '/home/gbetondji/smoothtransformer/data/imagewoof2-320/val'
valc_path = '/home/gbetondji/smoothtransformer/data/imagewoof2-320/val_c'


test_transforms_list = [
    transforms.Resize((size, size)),
    transforms.transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms_list = [
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    #transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

if augment:
    train_transforms = transforms.transforms.Compose(train_transforms_list)
else:
    train_transforms = test_transforms

test_transforms = transforms.transforms.Compose(test_transforms_list)



trainset = ImageFolder(
    root=train_path, 
    transform=train_transforms
)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)


valset = ImageFolder(
    root=val_path, 
    transform=train_transforms
)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

valcset = ImageFolder(
    root=valc_path, 
    transform=train_transforms
)
valc_loader = torch.utils.data.DataLoader(valcset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)


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
    dim = 512,
    depth = 16,
    heads = 16,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
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

    valc_loss = valc_acc = 0.0
    for (img, labels) in valc_loader:
        img, labels = img.to(device), labels.to(device)
        with torch.no_grad():
            predictions = model(img)
            loss = criterion(predictions, labels)
            correct = torch.argmax(predictions.data, 1) == labels
        valc_loss += loss
        valc_acc += correct.sum()
    valc_loss /= len(valc_loader.dataset)
    valc_acc /= len(valc_loader.dataset)

    print()
    print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.6f}")
    print(f" Valid acc {val_acc:.3f}, Val loss {val_loss:.6f}")
    print(f" Cortd acc {valc_acc:.3f}, Val loss {valc_loss:.6f}")
    
print(f"Trained {config.method}")
