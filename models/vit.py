import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from torch.autograd import Function

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def grid(m,n):
    A = torch.zeros((m*n, m*n))
    for i in range(m):
        for j in range(n):
            idx = i * n + j  # index of current pixel
            if i > 0:
                # connect to top neighbor
                A[idx, idx - n] = 1
            if i < n - 1:
                # connect to bottom neighbor
                A[idx, idx + n] = 1
            if j > 0:
                # connect to left neighbor
                A[idx, idx - 1] = 1
            if j < n - 1:
                # connect to right neighbor
                A[idx, idx + 1] = 1

    A = A/A.sum(0)
    return A

def tv_subtracted_term(X, lambda_, epsilon, A):
    """
    Computes the total variation subtracted term for the Total Variation-Subtracted Dirichlet Energy gradient flow.
    X: tensor of shape (b, n, d), where b is the batch size, n is the number of nodes, and d is the node representation size
    lambda_: scalar, regularization parameter
    epsilon: scalar, smoothing parameter for Moreau-Yosida regularization
    A: tensor of shape (n, n), adjacency matrix of the graph
    """
    b, n, d = X.shape
    # Moreau-Yosida regularization of X
    X_epsilon = (X - epsilon * torch.matmul(A, X)) / (1 + epsilon)  # Shape (b, n, d)

    # Compute the gradient of u_epsilon
    grad_u_epsilon = torch.matmul(A, X_epsilon) - X_epsilon  # Shape (b, n, d)

    # Compute the norm of the gradient
    norm_grad_u_epsilon = torch.norm(grad_u_epsilon, dim=2, keepdim=True)  # Shape (b, n, 1)

    # Compute the divergence term
    div_term = torch.matmul(A.transpose(-1,-2), grad_u_epsilon / (norm_grad_u_epsilon + 1e-10))  # Shape (b, n, d)

    return lambda_ * div_term

def diffusivity(M,m=4,k=0.1,l=-3.31488):
    return 1-torch.exp(l/torch.pow(M/k,m))

class CustomSign(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()  # Forward pass is just sign function

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Backward pass is the gradient of the tanh function
        return (1 - torch.tanh(input)**2) * grad_input

def gaussian_kernel(size: int, sigma: float):
    """Create a Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid((x, y))

    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel = kernel / kernel.sum()

    return kernel

# classes

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., weight_norm=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.ModuleList([nn.Linear(dim, inner_dim, bias = False) for _ in range(2)]+[nn.utils.weight_norm(nn.Linear(dim, inner_dim, bias = False)) if weight_norm else nn.Linear(dim, inner_dim, bias = False)])
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, diff=1):
        x = self.norm(x)

        qkv = [func(x) for func in self.to_qkv]
        #self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(diff*attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=1, dropout = 0., weight_norm=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.tau*attn(x) + x
            x = self.tau*ff(x) + x
        return x
    
class TransformerFT(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.tau*attn(x) + (1-self.tau)*x
            x = self.tau*ff(x) + (1-self.tau)*x
        return x
    
class TransformerD(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        self.grid = kwargs['grid']
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.tau*attn(x) + x
            x = x - tv_subtracted_term(x, self.tau, 1e-10, self.grid.to(x.device))
            x = self.tau*ff(x) + x
            x = x - tv_subtracted_term(x, self.tau, 1e-10, self.grid.to(x.device))
        return x
    
class TransformerBS(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x_in):
        for attn, ff in self.layers:
            B = attn(x_in)
            S = attn(B+x_in)
            x = self.tau*(B-S) + x_in
            x = self.tau*ff(x) + x
        return x
    
class TransformerPM(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        K=1
        for attn, ff in self.layers:
            grad = attn(x)
            D = torch.exp(-(grad**2) / (K**2)) #diffusivity(grad)
            x = self.tau*D*grad + x
            grad = ff(x)
            D = torch.exp(-(grad**2) / (K**2)) #diffusivity(grad)
            x = self.tau*D*grad + x
        return x
    
class TransformerSF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        self.sign = CustomSign.apply
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        K=1
        for attn, ff in self.layers:
            grad = attn(x)
            D = self.sign(grad) #diffusivity(grad)
            x = self.tau*D*grad + x
            grad = ff(x)
            D = self.sign(grad) #diffusivity(grad)
            x = self.tau*D*grad + x
        return x
    
class TransformerPMK(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
        self.K1 = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for _ in range(depth)])
        self.K2 = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for _ in range(depth)])
    def forward(self, x):
        for idx, (attn, ff) in enumerate(self.layers):
            grad = attn(x)
            D = torch.exp(-(grad**2) / (self.K1[idx]**2)) #diffusivity(grad)
            x = self.tau*D*grad + x
            grad = ff(x)
            D = torch.exp(-(grad**2) / (self.K2[idx]**2)) #diffusivity(grad)
            x = self.tau*D*grad + x
        return x

class TransformerPMS(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=0.5, dropout = 0., weight_norm=True, **kwargs):
        super().__init__()
        self.ks=3
        self.layers = nn.ModuleList([])
        self.tau = tau
        self.patch_dims = kwargs['patch_dims']
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
        self.kernel = gaussian_kernel(self.ks,0.1).unsqueeze(0).unsqueeze(0).repeat(dim_head*heads, dim_head*heads, 1, 1)
        self.conv = lambda input, kernel, ks : nn.functional.conv2d(input, kernel.to(input.device), stride=1, padding=ks//2)
        #self.conv2 = nn.Conv2d(in_channels=dim_head*heads, out_channels=dim_head*heads, kernel_size=ks, stride=1, padding=ks//2, bias=False)
    def forward(self, x):
        K=1
        b,N,dim = x.shape
        H,W = self.patch_dims
        for attn, ff in self.layers:
            grad = attn(x)
            sgrad = self.conv(grad.view(b,H,W,dim).permute(0, 3, 1, 2),self.kernel,self.ks).permute(0, 2, 3, 1).view(b,N,dim)
            D = torch.exp(-(sgrad**2) / (K**2)) #diffusivity(grad)
            x = self.tau*D*grad + x
            grad = ff(x)
            sgrad = self.conv(grad.view(b,H,W,dim).permute(0, 3, 1, 2),self.kernel,self.ks).permute(0, 2, 3, 1).view(b,N,dim)
            D = torch.exp(-(sgrad**2) / (K**2)) #diffusivity(grad)
            x = self.tau*D*grad + x
        return x
    
class TransformerGDM(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=1, dropout = 0., weight_norm=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        beta = 0.5
        mgrad = 0
        for attn, ff in self.layers:
            grad = beta*mgrad + self.tau*attn(x)
            x = x + beta*mgrad + (1+beta)*grad
            mgrad = grad
            grad = beta*mgrad + self.tau*ff(x)
            x = x + beta*mgrad + (1+beta)*grad
            mgrad = grad
        return x

class TransformerHGD(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, tau=1, dropout = 0., weight_norm=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.tau = tau
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight_norm=weight_norm),
                nn.utils.weight_norm(nn.Linear(dim, dim)) if weight_norm else nn.Linear(dim, dim)
            ]))
    def forward(self, x):
        t = self.tau
        grad = torch.ones(x.shape, device=x.device)
        hlr = 1e-8
        for attn, ff in self.layers:
            grad, prev_grad = attn(x), grad
            t = t + hlr*torch.mean(grad*prev_grad)
            x = x + t*grad
            
            grad, prev_grad = ff(x), grad
            t = t + hlr*torch.mean(grad*prev_grad)
            x = x + t*grad
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., method='A',tau=1, weight_norm=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.grid = grid(image_height // patch_height,image_width // patch_width)

        methods = {
            'A':Transformer,
            'FT':TransformerFT,
            'D':TransformerD,
            'BS':TransformerBS,
            'PM':TransformerPM,
            'PMS':TransformerPMS,
            'PMK':TransformerPMK,
            'SF':TransformerSF
        }

        self.transformer = methods[method](dim, depth, heads, dim_head, tau, dropout, weight_norm, grid=self.grid, patch_dims = (image_height // patch_height,image_width // patch_width))

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    def ll_diffuse(self,img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)

        return x.cpu().detach().numpy()
    

class SimpleViTED(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., method='A',tau=1, weight_norm=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.grid = grid(image_height // patch_height,image_width // patch_width)

        methods = {
            'A':Transformer,
            'FT':TransformerFT,
            'D':TransformerD,
            'BS':TransformerBS,
            'PM':TransformerPM,
            'PMS':TransformerPMS,
            'PMK':TransformerPMK,
            'SF':TransformerSF
        }

        self.transformer = methods[method](dim, depth, heads, dim_head, tau, dropout, weight_norm, grid=self.grid, patch_dims = (image_height // patch_height,image_width // patch_width))
        self.decoder = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(dim, num_classes * patch_height * patch_width),
            Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2) ', p1 = patch_height, p2 = patch_width, c = num_classes),
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        return self.decoder(x)
    
    def ll_diffuse(self,img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)

        return x.cpu().detach().numpy()