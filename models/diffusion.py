import torch

import torch.nn as nn

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def diffusion_step(F,A,W,tau=1):
    return tau*torch.einsum('nkv,nvd -> nkd',A,torch.einsum('nvd,dk -> nvk',F,W)) + F

def diffusion_stepFT(F,A,W,tau=0.5):
    return tau*torch.einsum('nkv,nvd -> nkd',A,torch.einsum('nvd,dk -> nvk',F,W)) + (1-tau)*F

def diffusion_stepI(F,A,W,tau=1):
    return tau*torch.einsum('nkv,nvd -> nkd',A-torch.eye(A.size(1)),torch.einsum('nvd,dk -> nvk',F,W)) + F

def diffusion_stepIFT(F,A,W,tau=0.5):
    return tau*torch.einsum('nkv,nvd -> nkd',A-torch.eye(A.size(1)),torch.einsum('nvd,dk -> nvk',F,W)) + (1-tau)*F


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, depth, num_classes, sign=0, tau=1, weight_sharing=True, method='A', stdev=1, softw=False):
        super().__init__()
        model_bases = {
            'A':diffusion_step,
            'I':diffusion_stepI,
            'FT':diffusion_stepFT,
            'IFT':diffusion_stepIFT
        }
        self.weight_sharing = True
        self.sign = sign
        self.tau = tau
        self.step = model_bases[method]
        self.softw = softw
        self.attend = lambda F,K,Q,d: torch.nn.functional.softmax( torch.bmm(F@Q,(F@K).transpose(1,2)) * (d** -0.5) ,dim=-1)
        
        C,H,W = input_size
        self.dim = H*W
        
        self.depth = depth
        
        if self.weight_sharing:
            self.WVs = nn.ParameterList([nn.parameter.Parameter(torch.torch.normal(0,10*stdev,(C,C)))])
            self.WKs = nn.ParameterList([nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C)))])
            self.WQs = nn.ParameterList([nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C)))])
        else:
            self.WVs = nn.ParameterList(
                [ nn.parameter.Parameter(torch.torch.normal(0,10*stdev,(C,C))) for _ in range(self.depth) ]
            )
            self.WKs = nn.ParameterList(
                [ nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C))) for _ in range(self.depth) ]
            )
            self.WQs = nn.ParameterList(
                [ nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C))) for _ in range(self.depth) ]
            )
        
        self.last_layer = nn.Linear(self.dim,num_classes)
        
    def forward(self, image):
        X = image.flatten(2).transpose(1,2)
        
        for step in range(self.depth):
            layer_idx = min(step,len(self.WVs)-1)
            WV, WK, WQ = self.WVs[layer_idx], self.WKs[layer_idx], self.WQs[layer_idx]
            if self.sign != 0:
                W = 0.5*(WV+WV.T)
                W = self.sign * W@W.T
            else:
                W = WV
            A = self.attend(X,WK,WQ,self.dim)
            X = self.step(X,A,W,self.tau)

        pred = self.last_layer(X.mean(-1))
        return pred
        
    def diffuse(self, image):
        X0 = image.flatten(2).transpose(1,2)
        outputs = [X0]
        As = []
        
        for step in range(self.depth):
            layer_idx = min(step,len(self.WVs)-1)
            WV, WK, WQ = self.WVs[layer_idx], self.WKs[layer_idx], self.WQs[layer_idx]
            if self.sign != 0:
                W = 0.5*(WV+WV.T)
                W = self.sign * W@W.T
            else:
                W = WV
            if self.softw: 
                W = torch.nn.functional.softmax(W, -1)
            A = self.attend(outputs[-1],WK,WQ,self.dim)
            As.append(A)
            outputs.append(self.step(outputs[-1],A,W,self.tau))
            
        outputs = [x.cpu().detach().numpy() for x in outputs]
        As = [x.cpu().detach().numpy() for x in As]
        
        return outputs, As