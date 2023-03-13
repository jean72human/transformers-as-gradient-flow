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


class SimpleTransformer:
    def __init__(self, input_size, depth, num_classes, sign=0, tau=1, method='A', stdev=1):
        model_bases = {
            'A':diffusion_step,
            'I':diffusion_stepI,
            'FT':diffusion_stepFT,
            'IFT':diffusion_stepIFT
        }
        self.sign = sign
        self.tau = tau
        self.step = model_bases[method]
        self.attend = lambda F,K,Q,d: torch.nn.functional.softmax( torch.bmm(F@Q,(F@K).transpose(1,2)) * (d** -0.5) )
        
        C,H,W = input_size
        self.dim = H*W
        
        self.WVs = nn.ParameterList(
            [ nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C))) for _ in range(depth) ]
        )
        self.WKs = nn.ParameterList(
            [ nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C))) for _ in range(depth) ]
        )
        self.WQs = nn.ParameterList(
            [ nn.parameter.Parameter(torch.torch.normal(0,stdev,(C,C))) for _ in range(depth) ]
        )
        
        self.last_layer = nn.Linear(self.dim,num_classes)
        
    def forward(self, image):
        X = image.flatten(2).transpose(1,2)
        
        for WV, WK, WQ in zip(self.WVs, self.WKs, self.WQs):
            A = self.attend(X,WK,WV,self.dim)
            X = self.step(X,A,WV,self.tau)
        
        pred = self.last_layer(X.mean(-1))
        return pred
        
    def diffuse(self, image):
        X0 = image.flatten(2).transpose(1,2)
        outputs = [X0]
        As = []
        
        for WV, WK, WQ in zip(self.WVs, self.WKs, self.WQs):
            if self.sign != 0:
                W = 0.5*(WV+WV.T)
                W = self.sign * W@W.T
            else:
                W = WV
            A = self.attend(outputs[-1],WK,WQ,self.dim)
            As.append(A)
            outputs.append(self.step(outputs[-1],A,W,self.tau))
        
        return outputs, As