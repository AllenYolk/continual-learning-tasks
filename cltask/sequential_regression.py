from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt


class FuncRegressionDataSet(data.Dataset):

    def __init__(
        self, fx: Callable, start: float, end: float, dx: float = 0.02, 
        noise_sd: float = 0.
    ):
        self.x = torch.arange(start=start, end=end, step=dx)
        self.x = self.x.unsqueeze(dim=1)
        self.y = fx(self.x) + torch.randn_like(self.x) * noise_sd

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def get_sequential_regression_loader(
    fx: Callable, ranges: Sequence[Sequence[float]], dx: float = 0.02,
    noise_sd: float = 0., batch_size: int = 64, 
):
    """Get data-loaders for a sequential regression task.

    See this paper for a detailed description of the task: 
    Camp, B., Mandivarapu, J. K., & Estrada, R. (2020). Continual Learning with 
    Deep Artificial Neurons (arXiv:2011.07035). arXiv. 
    https://doi.org/10.48550/arXiv.2011.07035

    Args:
        fx (Callable): the target function.
        ranges (Sequence[Sequence[float]]): a list whose elements are lists like 
            `[min, max]`, representing the range of independent variable x in 
            each of the phases of the training procedure. 
        dx (float, optional): the gap between each pair of the adjacent point of
            independent variable x. Defaults to 0.02.
        noise_sd (float, optional): the standard deviation of white noise. 
            Defaults to 0..
        batch_size (int, optional): a parameter for DataLoader. Defaults to 64.

    Returns:
        train_loaders (List[DataLoader]): a list of data-loaders for training.
            Each one is for a single phase in the training procedure. On each
            iteration, DataLoader in train_loaders gives out a tensor with shape
            [batch_size, 1].
        test_loader (DataLoader): a single data-loader for the overall test. On
            each iteration, test_loader gives out a tensor with shape 
            [batch_size, 1].
    """
    train_loaders = []
    xx_min, xx_max = float("inf"), -float("inf")
    for x_min, x_max in ranges:
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        train_loaders.append(data.DataLoader(
            dataset=FuncRegressionDataSet(fx, x_min, x_max, dx, noise_sd),
            batch_size=batch_size, shuffle=True, drop_last=False
        ))
        xx_min = min(xx_min, x_min)
        xx_max = max(xx_max, x_max)
    test_loader = data.DataLoader(
        dataset=FuncRegressionDataSet(fx, xx_min, xx_max, dx, noise_sd),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    return train_loaders, test_loader


def plot_sequential_regression(
    model: nn.Module, fx: Callable, start: float, end: float, dx: float = 0.02,
):
    x = torch.arange(start, end, dx).unsqueeze(dim=1)
    y = fx(x).squeeze()
    pred = model(x).squeeze()
    x = x.squeeze(x)

    plt.style.use("ggplot")
    _, ax = plt.subplots()
    ax.plot(x, y, label="y")
    ax.plot(x, pred, label="pred")
    ax.legend()
    plt.show()
