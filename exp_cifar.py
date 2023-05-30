import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import torch.optim as optim
import numpy as np

from utils import train, get_similarities

from models.diffusion import SimpleTransformer

import wandb

import os

batch_size = 128
size = (3,32,32)
patch_size = 2
depth = 64
device = 'cuda:0'
data_path = './data/'
num_epochs = 50
learning_rate = 1e-4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ms = ['PME','PMR']
ss = [0,0]
wns = [False,False]
ts = [1,1]
ws = False

with open('key.txt') as f:
   wandbkey = f.read()

os.environ["WANDB_API_KEY"] = wandbkey

for m,s,wn,t in zip(ms,ss,wns,ts):
    run = wandb.init(
        project="tgf",
        reinit=True,
        config={
            "model":m,
            "sign":s,
            "weight norm":wn,
            "tau":t
        }
    )
    model = SimpleTransformer(size, patch_size, depth, 
        dim=128, 
        heads=1,
        num_classes=10, 
        sign=s, 
        tau=t, 
        embed=True,
        weight_sharing=ws, 
        method=m,
        norm=True,
        attn_norm=False,
        weight_norm=wn,
    )
    
    model = model.to(device)

    print(f"model_method_{m}_sign_{s}_tau_{t}_wn_{wn}_ws_{ws}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accs_list = []
    train_loss_list = []
    val_accs_list = []
    val_loss_list = []

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
        val_loss /= batch_size*len(val_loader)
        val_acc /= batch_size*len(val_loader)
        
        train_accs_list.append(train_acc)
        val_accs_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        ## get dirichlet energy
        o_list, _, _ = model.diffuse(img)
        dirichlet_input = [x/(np.linalg.norm(x, ord='fro', axis=(-1,-2))) for x in o_list]
        dir_energy = get_similarities(dirichlet_input)[-1]

        wandb.log({"train_acc": train_acc,
                   "train_loss": train_loss,
                   "valid_acc": val_acc,
                   "val_loss": val_loss,
                   "dir_energy": dir_energy})
        
        if epoch%5==0 or epoch==num_epochs or epoch==1:
            print()
            print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.6f}")
            print(f" Valid acc {val_acc:.3f}, Val loss {val_loss:.6f}")
    run.finish()

    torch.save(model.state_dict(),f"model_method_{m}_sign_{s}_tau_{t}_wn_{wn}_ws_{ws}_acc_{round(100*val_acc.item())}.pth")
