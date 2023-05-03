import torch

import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch.nn import init

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class SignFunctionSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
class SignSTE(nn.Module):
    def forward(self, input):
        return SignFunctionSTE.apply(input)


def diffusion_step(F,A,W,heads,tau=1):
    return tau*torch.bmm(A,rearrange(F@W, 'b h n d -> b n (h d)', h = heads)) + rearrange(F, 'b h n d -> b n (h d)')

def diffusion_stepD(F,A,W,heads,tau=1):
    X = diffusion_step(F,A,W,heads,tau) 
    alpha = 2
    return X - tau*alpha*torch.tanh(alpha*X.sum())#tau*SignFunctionSTE.apply(X.sum())

def diffusion_stepN(F,A,W,heads,tau=1):
    X = diffusion_step(F,A,W,heads,tau) 
    return X * (1+torch.linalg.norm(X))

def diffusion_stepFT(F,A,W,heads,tau=0.5):
    return tau*torch.bmm(A,rearrange(F@W, 'b h n d -> b n (h d)', h = heads)) + (1-tau)*rearrange(F, 'b h n d -> b n (h d)')

def diffusion_stepI(F,A,W,tau=1):
    return tau*torch.bmm(A-torch.eye(A.size(1)),rearrange(F@W, 'b h n d -> b n (h d)', h = heads)) + rearrange(F, 'b h n d -> b n (h d)')

def diffusion_stepIFT(F,A,W,tau=0.5):
    return tau*torch.bmm(A-torch.eye(A.size(1)),rearrange(F@W, 'b h n d -> b n (h d)', h = heads)) + (1-tau)*rearrange(F, 'b h n d -> b n (h d)')

#def diffusion_stepTV(F,A,W,heads,tau=1):
#    S = rearrange(torch.sign(-torch.bmm(A,rearrange(F@W, 'b h n d -> b n (h d)', h = heads))), 'b n (h d) -> b h n d', h = heads)
#    return tau*torch.bmm(A.transpose(-2,-1),rearrange(S@W.T, 'b h n d -> b n (h d)', h = heads)) + rearrange(F, 'b h n d -> b n (h d)')

def diffusion_stepTV(F,A,W,heads,tau=1):
    S = rearrange(torch.tanh(-torch.bmm(A,rearrange(F@W, 'b h n d -> b n (h d)', h = heads))), 'b n (h d) -> b h n d', h = heads)
    return tau*torch.bmm(A.transpose(-2,-1),rearrange(S@W.T, 'b h n d -> b n (h d)', h = heads)) + rearrange(F, 'b h n d -> b n (h d)')

def diffusion_stepTVL2(F,A,W,heads,tau=1):
    S = rearrange(torch.tanh(-torch.bmm(A,rearrange(F@W, 'b h n d -> b n (h d)', h = heads))), 'b n (h d) -> b h n d', h = heads)
    return tau*torch.bmm(A.transpose(-2,-1),rearrange(S@W.T, 'b h n d -> b n (h d)', h = heads)) + (1-tau)*rearrange(F, 'b h n d -> b n (h d)')

def diffusion_stepQL(F,A,W,heads,tau=1):
    G = F@W
    L = torch.eye(A.size(1), device=A.device) - A
    term1 = -(L.transpose(-1,-2)+L)
    term2 = torch.tanh(torch.bmm(torch.bmm(rearrange(G, 'b h n d -> b n (h d)', h = heads).transpose(-1,-2),L),rearrange(G, 'b h n d -> b n (h d)', h = heads)))
    return rearrange(F, 'b h n d -> b n (h d)') - tau* rearrange(rearrange(torch.bmm(torch.bmm(term1,rearrange(G, 'b h n d -> b n (h d)', h = heads)),term2), 'b n (h d) -> b h n d', h = heads)@W.T, 'b h n d -> b n (h d)')


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, patch_size, depth, dim=1024, heads=9, num_classes=10, sign=0, tau=1, weight_sharing=True, method='A', embed=True, softw=False, norm=True, weight_norm=False, attn_norm=False, identities=False):
        super().__init__()
        model_bases = {
            'A':diffusion_step,
            'M':diffusion_step,
            'I':diffusion_stepI,
            'FT':diffusion_stepFT,
            'IFT':diffusion_stepIFT,
            'TV': diffusion_stepTV,
            'TVL2': diffusion_stepTVL2,
            'QL': diffusion_stepQL,
            'D': diffusion_stepD
        }
        self.weight_sharing = weight_sharing
        self.weight_norm = weight_norm
        self.sign = sign
        self.tau = tau
        self.step = model_bases[method]
        self.softw = softw
        self.normalize = norm
        self.attn_norm = attn_norm
        self.heads = heads
        self.identities = identities
        
        C,H,W = input_size
        num_patches = (H // patch_size) * (W // patch_size)
        
        self.depth = depth
        
        patch_dim = C * patch_size * patch_size
        self.dim = dim*heads
        self.vdim = dim
        if embed:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, self.dim),
                nn.LayerNorm(self.dim),
            )
        else:
            self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
            self.dim = patch_dim
        
        if self.weight_sharing:
            self.WVs = nn.ParameterList([nn.parameter.Parameter(init.kaiming_uniform_(torch.empty((self.vdim,self.vdim))))])
            self.WKs = nn.ParameterList([nn.parameter.Parameter(init.kaiming_uniform_(torch.empty((self.dim,self.dim))))])
            self.WQs = nn.ParameterList([nn.parameter.Parameter(init.kaiming_uniform_(torch.empty((self.dim,self.dim))))])
            self.norms = nn.ParameterList([nn.LayerNorm(self.dim)])
        else:
            self.WVs = nn.ParameterList(
                [ nn.parameter.Parameter(init.kaiming_uniform_(torch.empty((self.vdim,self.vdim)))) for _ in range(self.depth) ]
            )
            self.WKs = nn.ParameterList(
                [ nn.parameter.Parameter(init.kaiming_uniform_(torch.empty((self.dim,self.dim)))) for _ in range(self.depth) ]
            )
            self.WQs = nn.ParameterList(
                [ nn.parameter.Parameter(init.kaiming_uniform_(torch.empty((self.dim,self.dim)))) for _ in range(self.depth) ]
            )
            self.norms = nn.ParameterList(
                [ nn.LayerNorm(self.dim) for _ in range(self.depth) ]
            )
        
        self.last_layer = nn.Linear(self.dim,num_classes)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        
    def attend(self,F,K,Q,d): 
        attn_logits = torch.bmm(F@Q,(F@K).transpose(-2,-1)) * (d** -0.5) 
        if self.attn_norm:
            attn_logits = attn_logits/torch.linalg.norm(attn_logits)
        return torch.nn.functional.softmax( attn_logits ,dim=-1)
        
    def forward(self, image):
        X = self.to_patch_embedding(image)
        X += self.pos_embedding[:, :(X.size(1))]
        
        for step in range(self.depth):
            if self.training:
                X = nn.functional.dropout(X,0.1)
            layer_idx = min(step,len(self.WVs)-1) 
            if self.normalize:
                X = self.norms[layer_idx](X)
            WV, WK, WQ = self.WVs[layer_idx], self.WKs[layer_idx], self.WQs[layer_idx]
            if self.sign != 0:
                W = 0.5*(WV+WV.T)
                W = self.sign * W@W.T
            else:
                W = WV
            if self.weight_norm:
                W = W/torch.linalg.norm(W)
            if self.softw: 
                W = torch.nn.functional.softmax(W, -1) * (1/self.vdim)

            if self.identities == True and step%2==1:
                A = torch.eye(X.size(-2), device=X.device)
            else:
                A = self.attend(X,WK,WQ,self.vdim)
                
            X = rearrange(X, 'b n (h d) -> b h n d', h = self.heads)
            X = self.step(X,A,W,self.heads,self.tau)
            #X = nn.functional.relu(X)

        pred = self.last_layer(X.mean(dim=1))
        return pred
        
    def diffuse(self, image):
        X0 = self.to_patch_embedding(image)
        outputs = [X0 + self.pos_embedding[:, :(X0.size(1))]]
        As = []
        Ws = []
        
        for step in range(self.depth):
            X = outputs[-1]
            layer_idx = min(step,len(self.WVs)-1)
            if self.normalize:
                X = self.norms[layer_idx](X)
            WV, WK, WQ = self.WVs[layer_idx], self.WKs[layer_idx], self.WQs[layer_idx]
            if self.sign != 0:
                W = 0.5*(WV+WV.T)
                W = self.sign * W@W.T
            else:
                W = WV
            if self.weight_norm:
                W = W/torch.linalg.norm(W)
            if self.softw: 
                W = torch.nn.functional.softmax(W, -1) * (1/self.vdim)
            Ws.append(W)

            # compute attentions
            if self.identities == True and step%2==1:
                A = torch.eye(X.size(-2), device=X.device)
            else:
                A = self.attend(X,WK,WQ,self.vdim)

            X = rearrange(X, 'b n (h d) -> b h n d', h = self.heads)
            X = self.step(X,A,W,self.heads,self.tau)
            As.append(A)
            outputs.append(X)
            
        outputs = [x.cpu().detach().numpy() for x in outputs]
        As = [x.cpu().detach().numpy() for x in As]
        
        return outputs, As, Ws
