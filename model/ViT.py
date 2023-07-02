import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    '''Self Attention
    Args:
        dim_hidden (int): dimention of hidden layer
        num_heads (int): number of mutlit head attention
        qkv_bias (bool): bias of linear layer
    '''
    def __init__(self, dim_hidden: int, num_heads: int, qkv_bias: bool=False):
        super().__init__()
        
        assert dim_hidden % num_heads == 0
        
        self.num_heads = num_heads
        dim_head = dim_hidden // num_heads
        
        # scale of softmax
        self.scale = dim_hidden ** (-0.5)
        
        # linear layer for qkv
        self.proj_in = nn.Linear(dim_hidden, dim_hidden * 3, bias=qkv_bias)
        
        # linear for feature
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)
        
    def forward(self, x: torch.Tensor):
        # Size: (batch size, num features, dim of features)
        batch_size, num_features = x.shape[:2]
        qkv = self.proj_in(x)
        
        # -> Size: (batch size, num features, QKV, num heads, dim of heads)
        # -> Size: (QKV, batch size, num heads, num features, dim of heads))
        qkv = qkv.view(batch_size, num_features, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)
        
        # Attention of q and k
        # Size: (batch size, num heads, num features, num features)
        attention = q.matmul(k.transpose(-2, -1))
        attention = (attention * self.scale).softmax(dim=-1)
        
        # collect value
        # Size: (batch size, num heads, num features, dim of heads)
        y = attention.matmul(v)
        
        # -> Size: (batch size, num features, num heads, dim of heads)
        y = y.permute(0, 2, 1, 3).flatten(2)
        
        y = self.proj_out(y)

        return y
    

class FNN(nn.Module):
    '''FNN in Transformer encoder
    Args:
        dim_hidden (int): input dim
        dim_feedforward (int): dim of feedforward
    '''
    def __init__(self, dim_hidden: int, dim_feedforward: int):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TransformerEncoder(nn.Module):
    '''Transformer Encoder
    Args:
        dim_hidden (int): input dim
        num_heads (int): num of heads
        dim_feedforward (int): dim of feedforward
    '''
    def __init__(self, dim_hidden: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        
        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)
        
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)
        
    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.norm2(x)
        x = self.fnn(x) + x
        
        return x
         

class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int, img_size: int,
                 patch_size: int, dim_hidden: int, num_heads: int,
                 dim_feedforward: int, num_layers: int):
        super().__init__()
        
        assert img_size % patch_size == 0
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        num_patches = (img_size // patch_size) ** 2
        dim_patch = 3 * patch_size ** 2
    
        self.patch_embed = nn.Linear(dim_patch, dim_hidden)
        
        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim_hidden)
        )
        
        # class embedding
        self.class_token = nn.Parameter(
            torch.zeros((1, 1, dim_hidden))
        )

        # Transformer encoder layer
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoder(dim_hidden, num_heads, dim_feedforward)
             for _ in range(num_layers)]
        )
        
        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)
        
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        batch_size, c, h, w = x.shape

        assert h == self.img_size and w == self.img_size
        
        # decompose to patches
        x = x.view(batch_size, c, h // self.patch_size, self.patch_size,
                   w // self.patch_size, self.patch_size)
        
        # Size: (batch_size, patch row, patch wcol, c, patch size, patch size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # faltten
        x = x.reshape(batch_size, (h // self.patch_size) * (w // self.patch_size), -1)
        
        x = self.patch_embed(x)
        
        class_token = self.class_token.expand(batch_size, -1, -1)
        
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embed
        
        # encoding
        for layer in self.encoder_layers:
            x = layer(x)
        
        # extract featuring
        x = x[:, 0]
        
        if return_embed:
            return x
        
        x = self.linear(x)
                
        return x   