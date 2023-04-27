import pickle
import argparse
import os

import numpy as np
from torch.utils import data
from torchvision import datasets, transforms


def prepare_split_benchmark(
    original_dataset, data_dir, n_subtask, n_categories_per_subtask, train=True
):
    data_bins = [[] for _ in range(n_subtask)]
    target_bins = [[] for _ in range(n_subtask)]
    for x, y in original_dataset:
        bid = y // n_categories_per_subtask
        data_bins[bid].append(x)
        target_bins[bid].append(y)

    folder = "train" if train else "test"
    dspath = os.path.join(data_dir, f"split-{n_subtask}/{folder}")
    if not os.path.exists(dspath):
        os.makedirs(dspath)
    for i in range(n_subtask):
        print(
            f"subtask {i}: N_data={len(data_bins[i])}, "
            f"N_target={len(target_bins[i])}"
        )
        x = data_bins[i]
        y = np.array(target_bins[i])
        fpath = os.path.join(dspath, f"{i}")
        with open(fpath, "wb+") as f:
            pickle.dump({"data": x, "targets": y}, f)


class SplitCIFAR100(data.Dataset):

    def __init__(
        self, root: str, n_subtask: int, subtask_id: int, 
        train: bool=True, transform=None, target_transform=None
    ):
        super().__init__()
        if 100 % n_subtask != 0:
            raise ValueError("100 should be dividable by n_subtask")
        if not os.path.exists(os.path.join(root, f"split-{n_subtask}")):
            c = 100 // n_subtask
            print(f"Prepare Split{n_subtask}CIFAR100......")
            ds = datasets.CIFAR100(root, train=True, download=True)
            prepare_split_benchmark(ds, root, n_subtask, c, train=True)
            ds = datasets.CIFAR100(root, train=False, download=True)
            prepare_split_benchmark(ds, root, n_subtask, c, train=False)
            print("Finish preparation!")

        folder = "train" if train else "test"
        fpath = os.path.join(root, f"split-{n_subtask}/{folder}/{subtask_id}")
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

    ds = SplitCIFAR100(args.data_dir, 10, 0, transform=transforms.ToTensor())
    dl = data.DataLoader(ds, 10, shuffle=True)
    bx, by = next(iter(dl))
    print(by)