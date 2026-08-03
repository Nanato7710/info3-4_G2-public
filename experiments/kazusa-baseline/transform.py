import torch
from torchvision.transforms import v2



def build_transform(img_size: int):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0),
        v2.RandAugment(interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32),
    ])

    return train_transform, val_transform