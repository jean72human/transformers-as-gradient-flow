import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def train(model, train_loader, optimizer, loss_func, epoch, device):

    """
    Trains and ERM classifier for an epoch and returns the train loss and train accuracy for that epoch
    """
    model.train()
    train_loss = train_acc = 0.0
    for (img, labels) in train_loader:
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(img)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == labels
        train_loss += loss
        train_acc += correct.sum()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    return train_loss, train_acc

def batch_dirichlet(X):
    bs = X.shape[0]
    dim = X.shape[1]
    
    laplacian = -np.ones((dim,dim))
    np.fill_diagonal(laplacian,dim-1)
    batch_laplacian = np.repeat(laplacian[None,:,:], bs, axis=0)
    
    to_trace = np.transpose(X,(0,2,1))@batch_laplacian@X
    return sum([np.trace(D) for D in to_trace])/(2*bs)


def get_similarities(outputs):
    similarities = []
    batch_size = outputs[0].shape[0]
    dim = outputs[0].shape[1]
    for output in outputs:
        similarities.append(batch_dirichlet(output))
    
    return similarities 

def plot_similarities(sims):
    plt.plot(sims, linestyle=':', marker='P', markersize=5)
    plt.ylabel("Average Dirichlet energy")
    plt.xlabel("Diffusion steps")
    plt.show()
    
def get_differences(outputs):
    differences = []
    for k in range(len(outputs)-1):
        differences.append(np.linalg.norm(outputs[k+1])/(np.linalg.norm(outputs[k])+1e-9))
        
    return differences 

def plot_differences(diffs):
    plt.plot(diffs, linestyle=':', marker='P', color='r', markersize=5)
    plt.ylabel("Norm of the step")
    plt.xlabel("Diffusion steps")
    plt.show()
    
def get_images(outputs):
    images = []
    for output in outputs:
        images.append(np.reshape(output,(output.shape[0],int(np.sqrt(output.shape[1])),int(np.sqrt(output.shape[1])),output.shape[-1])))
        
    return images

def plot_images(norm_images,n=10):
    f, axarr = plt.subplots(1,n, figsize=(100, 20)) 
    steps = [int((k/n)*len(norm_images)) for k in range(n)]
    for i,step in enumerate(steps):
        image = ((norm_images[step]+1)/2)[0]
        axarr[i].imshow(image, aspect='equal')
        
    plt.show()

def plot_images_full(norm_images):
    f, axarr = plt.subplots(1,len(norm_images), figsize=(100, 20)) 
    for i,norm_image in enumerate(norm_images):
        image = ((norm_image+1)/2)[0]
        axarr[i].imshow(image, aspect='equal')
        
    plt.show()
        
    
def generate_matrix(dim,sign=1):
    V = np.diag(sign*np.random.rand(dim))
    Q = np.random.normal(0,0.5,(dim,dim))
    M = Q@V@np.linalg.inv(Q)
    return 0.5*(M+M.T)
