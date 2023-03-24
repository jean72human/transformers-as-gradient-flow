import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import torch.optim as optim

from utils import train

from models.diffusion import SimpleTransformer

batch_size = 256
size = (3,32,32)
patch_size = 2
depth = 32
device = 'cuda:1'
data_path = './data/'
num_epochs = 100
learning_rate = 1e-4

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

for m in ['A']:
    for s in [0]:
        for ws in [False]:
            model = SimpleTransformer(size, patch_size, depth, 
                            dim=64, 
                            heads=1,
                            num_classes=10, 
                            sign=s, 
                            tau=1, 
                            embed=True,
                            softw=False,
                            weight_sharing=ws, 
                            method=m,
                            norm=True)
            
            model = model.to(device)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

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
                val_loss /= len(val_loader.dataset)
                val_acc /= len(val_loader.dataset)
                
                train_accs_list.append(train_acc)
                val_accs_list.append(val_acc)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                
                if epoch%5==0 or epoch==num_epochs or epoch==1:
                    print()
                    print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.6f}")
                    print(f" Valid acc {val_acc:.3f}, Val loss {val_loss:.6f}")


            torch.save(model.state_dict(),f"model_method_{m}_sign_{s}_ws_{ws}_acc_{round(100*val_acc.item())}.pth")