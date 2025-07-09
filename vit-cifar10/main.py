from parameters import Parameters as p

from model import VisionTransformer
from dataset import train_loader, test_loader


model = VisionTransformer(
    p.img_size, p.patch_size, p.channel, p.embed_dim, p.num_heads, p.mlp_dim, p.drop_rate, p.depth, p.num_classes
)

print(train_loader.__len__())
print(test_loader.__len__())