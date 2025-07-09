import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.embed_dim, # number of kernels in this layer
                              kernel_size=self.patch_size,
                              stride=self.patch_size)
        num_patches = (self.img_size//self.patch_size)**2
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1+num_patches, self.embed_dim))
        
    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        cls_token = self.cls_token(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x
    

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 drop_rate):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=in_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x))) #! what is gelu
        x = self.drop(self.fc2(x)) # ! we don't usually set a dropout in final layer
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm2(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, channel, embed_dim, num_heads, mlp_dim, drop_rate, depth, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, channel, embed_dim)
        self.encoder = nn.Sequential(*
            [TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Linear(in_features=embed_dim, out_features=num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:,0]
        return self.classification_head(cls_token)


