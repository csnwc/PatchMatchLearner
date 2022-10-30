# pairdatamodule.py
import os
from typing import Any, Callable, Optional
from tqdm import tqdm
import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST

import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    '''https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d'''
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ImageFolderWithShifts(ImageFolder):
    '''https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d'''
    def __init__(
        self,
        root: str, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        super().__init__(root, transform)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index): # this is what ImageFolder normally returns 

        original_tuple = super(ImageFolder, self).__getitem__(index)
        
        path = self.imgs[index][0] # the image file path

        mv = torch.randint(-1, 2, (2,)) # the shift (y,x)
        f = original_tuple[0]
        crop_height, crop_width = f.shape[1]-2, f.shape[2]-2
        
        f0 = torchvision.transforms.functional.crop(f, 1, 1, crop_height, crop_width)
        f1 = torchvision.transforms.functional.crop(f, mv[0]+1, mv[1]+1, crop_height, crop_width)
        
        shifted_pair = torch.stack(
            [
                transform_lib.functional.crop(f, 1, 1, crop_height, crop_width),
                transform_lib.functional.crop(f, mv[0]+1, mv[1]+1, crop_height, crop_width)
            ]
        ).permute((1, 0, 2, 3))

        tuple_with_path = ((shifted_pair,mv.to(torch.float32)) + original_tuple + (path,))
        return tuple_with_path

class PairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        predict_dir: Optional[str] = None,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 16,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.predict_dir = predict_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self) -> None:
        _prepare_MNIST_instance(self.data_dir)

    def setup(self, stage: Optional[str]=None) -> None:
        if stage == 'fit':
            self.dataset = ImageFolderWithShifts(os.path.join(self.data_dir, 'images'),
                    transform = transform_lib.Compose([
                        transform_lib.Grayscale(),
                        transform_lib.ToTensor(),
                        transform_lib.GaussianBlur(5, sigma=(1.0, 2.0))
                    ])
                )
            splits = [int(sv*len(self.dataset)) for sv in [0.6, 0.2, 0.2]]
            self.p_train, self.p_val, self.p_test = torch.utils.data.random_split(self.dataset, splits, generator=torch.Generator().manual_seed(42))
            print(f"{[len(ds) for ds in [self.p_train, self.p_val, self.p_test]]}")

        elif stage == 'validate':
            pass
        elif stage == 'test':
            pass
        elif stage == 'predict':
            self.p_predict = ImageFolderWithPaths(self.predict_dir,
                    transform = transform_lib.Compose([
                        transform_lib.Grayscale(),
                        transform_lib.ToTensor(),
                        # transform_lib.GaussianBlur(5, sigma=(1.0, 2.0)) ?
                    ])
                )
            print(f"predict dataset size: {len(self.p_predict)} images.")

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.p_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.p_val, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._data_loader(self.p_predict, shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            # collate_fn=_collate_fn,
        )

def _prepare_MNIST_instance(
        data_dir: str,
        batch_size: int=64
    ) -> None:

    if os.path.isdir(os.path.join(data_dir, "MNIST")):
        print(f"MNIST already install in {data_dir}")
        return

    mnist_test = MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform_lib.Compose([transform_lib.ToTensor()])
    )

    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    for class_idx in mnist_test.class_to_idx.values():
        os.makedirs(os.path.join(data_dir, 'images', str(class_idx)), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'montages'), exist_ok=True)

    data_loader = DataLoader(
            mnist_test,
            batch_size=batch_size,
            num_workers=0,
        )

    class_cnt = [0]*len(mnist_test.classes)
    for batch_num, (images, targets) in enumerate(tqdm(data_loader, desc="MNIST extract test")):

        for image, target in zip(images, targets):
            torchvision.utils.save_image(
                image,
                os.path.join(data_dir,
                    'images', str(target.item()),
                    f"{target}{class_cnt[target]:05d}.png"
                )
            )
            class_cnt[target] += 1

        torchvision.utils.save_image(
            torchvision.utils.make_grid(images, nrow=8),
            os.path.join(data_dir, 'montages', f"batch_{batch_num:04d}.png")
        )
