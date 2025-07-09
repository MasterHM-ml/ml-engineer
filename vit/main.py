from model import VisionTransformer

from parameters import Parameters as p

model = VisionTransformer(
    p.img_size, p.patch_size, p.channel, p.embed_dim, p.num_heads, p.mlp_dim, p.drop_rate, p.depth, p.num_classes
)