import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import torch.optim as optim
from torchvision.datasets import ImageFolder
import numpy as np

from utils import train, get_similarities

import timm

from models.vit import SimpleViT

import wandb

import os

augment = True
batch_size = 128
num_epochs = 50
size = 224
patch_size = 32
depth = 32
learning_rate = 1e-4
dropout = 0.2
device = 'cuda:1'
log = False
train_path = './data/imagewoof2-320/train'
val_path = './data/imagewoof2-320/val'
valc_path = './data/imagewoof2-320/val_c'


test_transforms_list = [
    transforms.Resize((size, size)),
    transforms.transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms_list = [
    transforms.Resize((size, size)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=dropout),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


test_transforms = transforms.transforms.Compose(test_transforms_list)

if augment:
    train_transforms = transforms.transforms.Compose(train_transforms_list)
else:
    train_transforms = test_transforms

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


ms = ['deit_g2_patch16_224']#,'deit_g2_patch16_224','deit_sheaf_patch16_224','deit_small_patch16_224','featscale_small_12','deit_g2_patch16_224']
ss = [0]
wns = [False]
ts = [1]
ws = False

if log:
    with open('key.txt') as f:
        wandbkey = f.read()

    os.environ["WANDB_API_KEY"] = wandbkey

for m,s,wn,t in zip(ms,ss,wns,ts):
    if log:
        run = wandb.init(
            project="tgf",
            reinit=True,
            config={
                "dataset":"imagewoof-augmented-pretrained",
                "model":m,
                "sign":s,
                "weight norm":wn,
                "tau":t,
                "depth":depth,
                "dropout":dropout
            }
        )
    model = timm.create_model(m, pretrained=False)
    
    model = model.to(device)

    print(f"model_method_{m}_sign_{s}_tau_{t}_wn_{wn}_ws_{ws}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accs_list = []
    train_loss_list = []
    val_accs_list = []
    val_loss_list = []
    valc_accs_list = []
    valc_loss_list = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):  
        train_loss, train_acc, train_dir = train(model,train_loader,optimizer,criterion,epoch,device,dir=True)  

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
        val_loss /= batch_size*len(val_loader)
        val_acc /= batch_size*len(val_loader)

        valc_loss = valc_acc = 0.0
        for (imgc, labels) in valc_loader:
            imgc, labels = imgc.to(device), labels.to(device)
            with torch.no_grad():
                predictions = model(imgc)
                loss = criterion(predictions, labels)
                correct = torch.argmax(predictions.data, 1) == labels
            valc_loss += loss
            valc_acc += correct.sum()
        valc_loss /= batch_size*len(valc_loader)
        valc_acc /= batch_size*len(valc_loader)
        
        train_accs_list.append(train_acc)
        val_accs_list.append(val_acc)
        valc_accs_list.append(valc_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        valc_loss_list.append(valc_loss)

        ## get dirichlet energy
        last_representation = model.ll_diffuse(img)
        dirichlet_input = [ last_representation/np.linalg.norm(last_representation, ord='fro', axis=(-1,-2))[:,None,None] ]
        val_dir = get_similarities(dirichlet_input)[-1]

        last_representation = model.ll_diffuse(imgc)
        dirichlet_input = [ last_representation/np.linalg.norm(last_representation, ord='fro', axis=(-1,-2))[:,None,None] ]
        valc_dir = get_similarities(dirichlet_input)[-1]

        if log:
            wandb.log({"train_acc": train_acc,
                    "train_loss": train_loss,
                    "valid_acc": val_acc,
                    "val_loss": val_loss,
                    "corr_acc": valc_acc,
                    "corr_loss": valc_loss,
                    "train_dir":train_dir,
                    "val_dir": val_dir,
                    "corr_dir": valc_dir})
        
        if epoch%5==0 or epoch==num_epochs or epoch==1:
            print()
            print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.6f}")
            print(f" Valid acc {val_acc:.3f}, Val loss {val_loss:.6f}")
            print(f" Corr acc {valc_acc:.3f}, Val loss {valc_loss:.6f}")

    if log: run.finish()

    torch.save(model.state_dict(),f"model_method_{m}_sign_{s}_tau_{t}_wn_{wn}_ws_{ws}_acc_{round(100*val_acc.item())}.pth")
