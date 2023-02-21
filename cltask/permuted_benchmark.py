from typing import Optional, Union

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class Permutation(nn.Module):
    """Change the pixel order of the given image tensor.

    The same pixel-wise permutation is applied as long as the attribute `seed`
    is fixed. Hence, this module can be placed before a network model to 
    simulate a permutation on the benchmark data.
    Notice: the input tensor `x` must be a image tensor with shape [..., H, W].

    Attributes:
        seed (int): the RNG seed used to generate random permutation.
    """

    def __init__(self, seed: Optional[int] = None):
        """The constructor for Permutation.

        Args:
            seed (int, optional): the RNG seed used to generate a random 
                permutation. Defaults to None, where `seed` is generated by
                torch.seed() implicitly.
        """
        super().__init__()
        self.seed = torch.seed() if seed is None else seed

    def forward(self, x: torch.Tensor):
        # x does not have the batch dimension; x.shape = [..., H, W]
        pre_shape, xshape = x.shape[0:-2], x.shape[-2:]
        x = x.reshape(*pre_shape, -1)
        torch.manual_seed(self.seed)
        perm = torch.randperm(x.shape[-1])
        x = x[..., perm]
        return x.reshape(*pre_shape, *xshape)


class PermutationWrappedNetwork(nn.Module):
    """A wrapper adding a Permutation to the front of a neural network.

    Add this wrapper to a neural network to simulate a permutation on the 
    dataset. 

    Attributes:
        net (nn.Module): the wrapped neural network.
        permutation (Permutation): the Permutation module preceding the neural
            network.
        permutation_seed (int): the RNG seed for Permutation.
        apply_permutation (bool): whether to apply the permutation.
    """

    def __init__(
        self, net: nn.Module, seed: Optional[int] = None, 
        apply_permutation: bool = True
    ):
        """The constructor for PermutationWrappedNetwork.

        Args:
            net (nn.Module): the wrapped neural network.
            seed (int, optional): the RNG seed for Permutation. Defaults to 
                None, where `seed` is generated implicitly by torch.seed().
            apply_permutation (bool, optional): whether to apply the 
                permutation. Defaults to True.
        """
        super().__init__()
        self.net = net
        self.permutation = Permutation(seed = seed)
        self.apply_permutation = apply_permutation

    @property
    def permutation_seed(self) -> int:
        return self.permutation.seed

    def change_permutation(
        self, perm: Union[Permutation, int] = None
    ) -> Permutation:
        """Change the permutation module.

        Args:
            perm (Union[Permutation, int], optional): the new `Permutation` 
                module. Defaults to None, where a new Permutation is generated 
                implicitly using torch.seed().

        Returns:
            the new permutation.
        """
        if perm is None:
            self.permutation = Permutation()
        elif isinstance(perm, int):
            self.permutation = Permutation(seed=perm)
        else:
            self.permutation = perm
        return self.permutation

    def forward(self, x: torch.Tensor):
        if self.apply_permutation:
            x = self.permutation(x)
        return self.net(x)


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
