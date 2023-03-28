import pickle
import argparse
import os

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms


def prepare_split10_cifar100(data_dir, train=True):
    ds = datasets.CIFAR100(
        root=data_dir, 
        train=train,
        download=True,
    )

    data_bins = [[] for _ in range(10)]
    target_bins = [[] for _ in range(10)]
    for x, y in ds:
        bid = y//10
        data_bins[bid].append(x)
        target_bins[bid].append(y)

    folder = "train" if train else "test"
    for i in range(10):
        print(
            f"subtask {i}: N_data={len(data_bins[i])}, "
            f"N_target={len(target_bins[i])}"
        )
        x = data_bins[i]
        y = np.array(target_bins[i])
        fpath = os.path.join(data_dir, f"cifar-100-split-10/{folder}/{i}")
        with open(fpath, "wb+") as f:
            pickle.dump({"data": x, "targets": y}, f)


class Split10CIFAR100(data.Dataset):

    def __init__(
        self, root: str, subtask: int, 
        train: bool=True, transform=None, target_transform=None
    ):
        super().__init__()
        if not os.path.exists(os.path.join(root, "cifar-100-split-10")):
            print("Prepare Split10CIFAR100......")
            prepare_split10_cifar100(root, True)
            prepare_split10_cifar100(root, False)
            print("Finish preparation!")

        folder = "train" if train else "test"
        fpath = os.path.join(root, f"cifar-100-split-10/{folder}/{subtask}")
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        self.data = d["data"]
        self.targets = d["targets"]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/export/home/data_allenyolk/CIFAR-100"
    )
    args = parser.parse_args()
    print(args)
    prepare_split10_cifar100(args.data_dir, train=True)
    prepare_split10_cifar100(args.data_dir, train=False)

    ds = Split10CIFAR100(args.data_dir, 0, transform=transforms.ToTensor())
    dl = data.DataLoader(ds, 10, shuffle=True)
    bx, by = next(iter(dl))
    print(by)