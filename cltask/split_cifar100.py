import pickle
import argparse
import os

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms


def prepare_split_cifar100(data_dir, train=True):
    ds = datasets.CIFAR100(
        root=data_dir, 
        train=train,
        download=True,
        transform=transforms.ToTensor()
    )
    print(len(ds))

    loader = data.DataLoader(ds, 10, shuffle=False)
    x, y = next(iter(loader))
    print(x.shape, y.shape)
    print(y)

    data_bins = [[] for i in range(10)]
    label_bins = [[] for i in range(10)]
    for x, y in ds:
        bid = y//10
        data_bins[bid].append(x)
        label_bins[bid].append(y)

    folder = "train" if train else "test"
    for i in range(10):
        print(
            f"subtask {i}: N_data={len(data_bins[i])}, "
            f"N_label={len(label_bins[i])}"
        )
        x = torch.stack(data_bins[i]).numpy()
        y = np.array(label_bins[i])
        print("\t", x.shape, y.shape)
        fpath = os.path.join(data_dir, f"cifar-100-split-10/{folder}/{i}")
        with open(fpath, "wb+") as f:
            pickle.dump({"data": x, "labels": y}, f)


class Split10CIFAR100(data.Dataset):

    def __init__(
        self, root: str, subtask: int, 
        train: bool=True, transform=None, target_transform=None
    ):
        super().__init__()
        folder = "train" if train else "test"
        fpath = os.path.join(root, f"cifar-100-split-10/{folder}/{subtask}")
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        self.data = d["data"]
        self.labels = d["labels"]
        print(self.data.shape, self.labels.shape)
        self.transform = transform
        self.target_transform = target_transform

    # TODO: add __len__() and __getitem__()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return data, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/export/home/data_allenyolk/CIFAR-100"
    )
    args = parser.parse_args()
    print(args)
    prepare_split_cifar100(args.data_dir, train=True)
    prepare_split_cifar100(args.data_dir, train=False)

    ds = Split10CIFAR100(args.data_dir, 0)
    dl = data.DataLoader(ds, 10, shuffle=True)
    bx, by = next(iter(dl))
    print(by)