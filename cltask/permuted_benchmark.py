from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class Permutation(nn.Module):
    """A `Transform` on Tensors that can change the pixel order.

    The same pixel-wise permutation is applied as long as the attribute `seed`
    is fixed. Hence, this Transform can be used to create a permuted benchmark.
    Notice: the input tensor `x` must be a image tensor with shape [C, H, W].

    Attributes:
        seed (int): the RNG seed used to generate random permutation.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = torch.seed() if seed is None else seed

    def forward(self, x: torch.Tensor):
        # x does not have the batch dimension; x.shape = [C, H, W]
        c, xshape = x.shape[0], x.shape[1:]
        x = x.reshape(c, -1)
        torch.manual_seed(self.seed)
        perm = torch.randperm(x.shape[1])
        x = x[:, perm]
        return x.reshape(c, *xshape)


def change_dataset_permutation(
    dataset: datasets.VisionDataset, 
    new_permutation: Optional[Permutation] = None
) -> Permutation:
    """Change the pixel order of every image sample in the dataset.

    The same pixel-wise permutation is applied to every image sample in the
    vision dataset. This function is implemented by appending a Permutation
    to the transform list of the VisionDataset. Thanks to the fine-grained
    conditions, we can ensure that there's no more than 1 Permutation transform
    applied to a certain dataset at the same time, and Permutation must be 
    place at the last position of the transform list.

    Args:
        dataset (datasets.VisionDataset): the vision dataset to be permuted.
        new_permutation (Optional[Permutation], optional): a new Permutation
            transform instance. Defaults to None, which will create a new random
            Permutation transform instance.

    Returns:
        new_permutation (Permutation): the Permutation instance used to 
            transform the dataset.
    """
    if new_permutation is None:
        new_permutation = Permutation()

    tf = dataset.transform
    if (tf is None) or isinstance(tf, Permutation):
        dataset.transform = new_permutation
    elif isinstance(tf, transforms.Compose):
        if isinstance(tf.transforms[-1], Permutation):
            dataset.transform.transforms[-1] = new_permutation
        else:
            dataset.transform.transforms.append(new_permutation)
    else:
        dataset.transform = transforms.Compose([tf, new_permutation])

    return new_permutation


def change_datasets_permutation(
    datasets: Sequence[datasets.VisionDataset],
    new_permutation: Optional[Permutation] = None
) -> Permutation:
    """Change the pixel order of several datasets in the same manner.

    The same pixel-wise permutation is applied to every image sample in each
    vision dataset. This function is implemented by appending a Permutation
    to the transform list of each VisionDataset. Thanks to the fine-grained
    conditions, we can ensure that there's no more than 1 Permutation transform
    applied to a certain dataset at the same time, and Permutation must be 
    place at the last position of the transform list.

    Args:
        datasets (Sequence[datasets.VisionDataset]): a list of vision datasets 
            to be permuted.
        new_permutation (Optional[Permutation], optional): a new Permutation
            transform instance. Defaults to None, which will create a new random
            Permutation transform instance.

    Returns:
        new_permutation (Permutation): the Permutation instance used to 
            transform the dataset.
    """
    if new_permutation is None:
        new_permutation = Permutation()

    for dataset in datasets:
        change_dataset_permutation(dataset, new_permutation)

    return new_permutation


def remove_dataset_permutation(dataset: datasets.VisionDataset):
    """Remove the pixel-wise permutation transform applied to the dataset.

    If there's a Permutation transform applied to `dataset`, remove it. If there
    is no Permutation applied to `dataset`, nothing will happen. Since there's
    no more than 1 Permutation applied at the same time, and Permutation must be
    placed at the last position of the dataset's transform list, we can check
    it easily.

    Args:
        dataset (datasets.VisionDataset): the target dataset.
    """
    # There's no more than 1 Permutation;
    # Permutation must be placed at the last position of the transforms list.
    tf = dataset.transform
    if isinstance(tf, Permutation):
        dataset.transform = None
    elif (isinstance(tf, transforms.Compose) and len(tf.transforms) > 0 and
        isinstance(tf.transforms[-1], Permutation)):
        if len(tf.transforms) == 1:
            dataset.transform = None
        else:
            dataset.transform.transforms.pop()


def plot_permuted_benchmark_result_matrix(test_loss_mat, test_acc_mat):
    f, ax = plt.subplots(ncols=2)
    N = test_loss_mat.shape[0]
    im0 = ax[0].imshow(test_loss_mat, cmap="Reds")
    f.colorbar(im0, ax=ax[0], label="loss")
    ax[0].set(
        title="test loss", xticks=np.arange(N), yticks=np.arange(N),
        xlabel="tested on", ylabel="have learned"
    )
    im1 = ax[1].imshow(test_acc_mat, cmap="Blues")
    f.colorbar(im1, ax=ax[1], label="acc")
    ax[1].set(
        title="test acc", xticks=np.arange(N), yticks=np.arange(N),
        xlabel="tested on", ylabel="have learned"
    )
    return f, ax
