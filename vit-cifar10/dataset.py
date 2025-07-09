from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader

from parameters import Parameters as p


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(0,0),
])

train_loader = DataLoader(
    dataset=datasets.CIFAR10(
        root="/home/hm/data/code-practice",
        download=True,
        transform=transform
        ),
    batch_size=p.batch_size,
    num_workers=2
)
                          
test_loader = DataLoader(
    dataset=datasets.CIFAR10(
        root="/home/hm/data/code-practice",
        train=False,
        download=True, 
        transform=transform
    ),
    batch_size=p.batch_size,
    num_workers=2
)

