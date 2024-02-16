import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from tqdm import tqdm
import numpy as np

from einops import rearrange

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim):      
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        
    def forward(self, x):
        '''
        Args: 
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, input_dim)
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.dim)
        q, k, v = qkv.unbind(2)
        # ...or q, k, v = [qkv[:, :, idx, :] for idx in range(3)]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return x
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=8):
        ### your code is here
        super().__init__()
        if dim % num_heads:
            raise ValueError('dim % num_heads != 0')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        ### your code is here
        '''
        Args: 
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, input_dim)
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv: 3 × B × num_heads × N × head_dim
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # attn: B × num_heads × N × N    v: B × num_heads × N × head_dim
        # B × num_heads × N × head_dim
        x = x.transpose(1, 2).reshape(B, N, C) 
        # B × N × (num_heads × head_dim)
        x = self.proj(x)
        return x
    
class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4,  # ratio between hidden_dim and input_dim in MLP
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttentionBlock(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), 
                                 act_layer(), 
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

def img2patches(img, patch_size=8):
    '''
    Args:
        img: (batch_size, c, h, w) Tensor
        
    Returns:
        (batch_size, num_patches, vectorized_patch) Tensor
    '''
    return rearrange(img, 'batch_size c (h ph) (w pw) -> batch_size (h w) (c ph pw)', 
                     ph=patch_size, pw=patch_size)

class ViT(nn.Module):
    def __init__(self,
                img_size=(224, 224),
                patch_size=16,
                in_chans=3,
                num_classes=10,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            ):
        super().__init__()
        self.patch_size = patch_size
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, act_layer, norm_layer) for _ in range(depth)
        ])
        self.patch_proj = nn.Linear(1 * patch_size * patch_size, embed_dim) #nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.embed_len = (img_size[0] * img_size[1]) // (patch_size * patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, embed_dim) * .02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        '''
        Args: 
            x: (batch_size, in_channels, img_size[0], img_size[1])
            
        Return:
            (batch_size, num_classes)
        '''
        x = img2patches(x, patch_size=self.patch_size)
        x = self.patch_proj(x)
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.blocks(x)
        x = x[:, 0, :]  # take CLS token
        return self.head(x)