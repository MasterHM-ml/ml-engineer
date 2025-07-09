from dataclasses import dataclass

@dataclass
class Parameters:
    batch_size = 128
    epoch = 10

    patch_size = 4
    lr = 3e-4
    img_size = 32
    channel = 3

    num_classes = 10

    embed_dim = 256

    num_heads = 8
    depth = 6

    mlp_dim = 512

    drop_rate = 0.1
