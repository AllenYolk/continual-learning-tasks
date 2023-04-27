import argparse
import os
import pickle

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms


def _prepare_split_benchmark(
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
            _prepare_split_benchmark(ds, root, n_subtask, c, train=True)
            ds = datasets.CIFAR100(root, train=False, download=True)
            _prepare_split_benchmark(ds, root, n_subtask, c, train=False)
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


class TinyImageNet200(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.root_dir = root
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if self.train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()
        with open(wnids_file, "r") as f:
            for entry in f:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r") as f:
            for entry in f:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (
                        words[1].strip("\n").split(",")
                    )[0]

    def _create_class_idx_dict_train(self):
        classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        classes = sorted(classes)
        num_images = 0
        for _, _, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images += 1

        self.len_dataset = num_images
        self.target_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_target_idx = {classes[i]: i for i in range(len(classes))}


    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()

        with open(val_annotations_file, "r") as f:
            for data in f:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(self.val_img_to_class)
        classes = sorted(list(set_of_classes))
        self.target_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_target_idx = {classes[i]: i for i in range(len(classes))}

    def _make_dataset(self, train=True):
        self.images = []
        if train:
            img_root_dir = self.train_dir
            list_of_dirs = [d for d in self.class_to_target_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]
        
        for d in list_of_dirs:
            dirs = os.path.join(img_root_dir, d)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for name in sorted(files):
                    if name.endswith(".JPEG"):
                        path = os.path.join(root, name)
                        if train:
                            item = (path, self.class_to_target_idx[d])
                        else:
                            item = (
                                path, 
                                self.class_to_target_idx[
                                    self.val_img_to_class[name]
                                ]
                            )
                        self.images.append(item)

    def get_label(self, idx):
        return [self.class_to_label[self.target_idx_to_class[idx]]]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, target = self.images[idx]
        with open(img_path, "rb") as f:
            img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SplitTinyImageNet200(data.Dataset):

    def __init__(
        self, root: str, n_subtask: int, subtask_id: int, 
        train: bool=True, transform=None, target_transform=None
    ):
        super().__init__()
        if 200 % n_subtask != 0:
            raise ValueError("200 should be dividable by n_subtask")
        if not os.path.exists(os.path.join(root, f"split-{n_subtask}")):
            c = 200 // n_subtask
            print(f"Prepare Split{n_subtask}TinyImageNet200......")
            ds = TinyImageNet200(root, train=True)
            _prepare_split_benchmark(ds, root, n_subtask, c, train=True)
            ds = TinyImageNet200(root, train=False)
            _prepare_split_benchmark(ds, root, n_subtask, c, train=False)
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
        "--data_dir", type=str, 
        default="/export/home/data_allenyolk/tiny-imagenet-200"
    )
    args = parser.parse_args()
    print(args)

    ds = SplitTinyImageNet200(
        args.data_dir, 20, 1, transform=transforms.ToTensor()
    )
    dl = data.DataLoader(ds, 10, shuffle=True)
    bx, by = next(iter(dl))
    print(by)