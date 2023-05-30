import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import torch.optim as optim
from torchvision.datasets import ImageFolder
import numpy as np

from utils import train, get_similarities

from models.vit import SimpleViT

import wandb

from spawrious import get_spawrious_dataset

import os

augment = True
batch_size = 128
num_epochs = 50
size = 256
patch_size = 32
depth = 32
learning_rate = 2e-5
dropout = 0.2
device = 'cuda:1'
log = True
path = './data/spawrious'
comb = "o2o_easy"


spawrious = get_spawrious_dataset(dataset_name=comb, root_dir=path)

trainset = spawrious.get_train_dataset()
valset = spawrious.get_test_dataset()

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

ms = ['A','PM']
ss = [0,0]
wns = [False,False]
ts = [1,1]
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
                "dataset":f"spawrious-{comb}",
                "model":m,
                "sign":s,
                "weight norm":wn,
                "tau":t,
                "depth":depth,
                "dropout":dropout
            }
        )
    model = SimpleViT(
        image_size = size,
        patch_size = patch_size,
        num_classes = 10,
        dim = 1024,
        depth = depth,
        heads = 16,
        dropout = dropout,
        method = m,
        tau=t,
        weight_norm = wn
    )
    
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
        
        train_accs_list.append(train_acc)
        val_accs_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        ## get dirichlet energy
        last_representation = model.ll_diffuse(img)
        dirichlet_input = [ last_representation/np.linalg.norm(last_representation, ord='fro', axis=(-1,-2))[:,None,None] ]
        val_dir = get_similarities(dirichlet_input)[-1]

        if log:
            wandb.log({"train_acc": train_acc,
                    "train_loss": train_loss,
                    "valid_acc": val_acc,
                    "val_loss": val_loss,
                    "train_dir":train_dir,
                    "val_dir": val_dir})
        
        if epoch%5==0 or epoch==num_epochs or epoch==1:
            print()
            print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.6f}")
            print(f" Valid acc {val_acc:.3f}, Val loss {val_loss:.6f}")

    if log: run.finish()

    torch.save(model.state_dict(),f"model_method_{m}_sign_{s}_tau_{t}_wn_{wn}_ws_{ws}_acc_{round(100*val_acc.item())}.pth")
