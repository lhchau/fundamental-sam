from .cifar10 import get_cifar10
from .cifar100 import get_cifar100
from .tiny_imagenet import get_tiny_imagenet


# Data
def get_dataloader(
    dataset='cifar10',
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    print('==> Preparing data..')

    if dataset == "cifar10":
        return get_cifar10(
            batch_size,
            num_workers,
            split
        )
    elif dataset == "cifar100":
        return get_cifar100(
            batch_size,
            num_workers,
            split
        )
    elif dataset == "tiny_imagenet":
        return get_tiny_imagenet(
            batch_size,
            num_workers
        )

