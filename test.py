import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

import cltask
import reunn


def f_target(x):
        return torch.sin(x*2+0.3) + 0.5*torch.sin(x*3-0.2)


def sequential_regression_overall_train(epochs: int, device):
    net = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    st = reunn.NetStats(net=net, input_shape=[1, 1])
    st.print_summary()

    ranges = [[x, x+2] for x in range(-5, 5, 2)]
    _, _, overall_loader = cltask.get_sequential_regression_loader(
        fx=f_target, ranges=ranges, batch_size=100
    )

    cltask.plot_sequential_regression(
        model=net, fx=f_target, start=-5, end=5
    )
    plt.show()
    p = reunn.SupervisedTaskPipeline(
            backend="torch", log_dir="../log_dir", net=net, 
            hparam=None, device=device,
            criterion=nn.MSELoss(), 
            optimizer=optim.Adam(net.parameters(), lr=1e-3),
            train_loader=overall_loader,
        )
    p.train(epochs=epochs, silent=True)
    cltask.plot_sequential_regression(
        model=net, fx=f_target, start=-5, end=5
    )
    plt.show()


def sequential_regression_phasic_train(epochs: int, device):
    net = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    ranges = [[x, x+2] for x in range(-5, 5, 2)]
    train_loaders, test_loaders, overall_loader =\
        cltask.get_sequential_regression_loader(
            fx=f_target, ranges=ranges, batch_size=100
        )

    phases = len(ranges)
    cltask.plot_sequential_regression(
        model=net, fx=f_target, start=-5, end=5, #title="original"
        fill_x_ranges=ranges
    )
    plt.show()
    for phase in range(phases):
        p = reunn.SupervisedTaskPipeline(
            backend="torch", log_dir="../log_dir", net=net, 
            hparam=None, device=device,
            criterion=nn.MSELoss(), 
            optimizer=optim.Adam(net.parameters(), lr=1e-3),
            train_loader=train_loaders[phase],
            validation_loader=test_loaders[phase],
            test_loader=test_loaders[phase]
        )
        p.train(epochs=epochs, validation=True, silent=True)
        p.test()
        _, ax = cltask.plot_sequential_regression(
            model=net, fx=f_target, start=-5, end=5, dx=0.02,
            fill_x_ranges=[ranges[phase]]
        )
        ax.set_title(f"phase {phase}: {ranges[phase]}")
        plt.show()


def permuted_mnist(n_subtask, epochs, device):
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    net = cltask.PermutationWrappedNetwork(net, apply_permutation=True)

    train_loader = data.DataLoader(
        dataset=datasets.MNIST(
            root="../datasets", download=True, train=True,
            transform=transforms.ToTensor()
        ),
        batch_size=64, shuffle=False, drop_last=False
    )
    test_loader = data.DataLoader(
        dataset=datasets.MNIST(
            root="../datasets", download=True, train=False, 
            transform=transforms.ToTensor()
        ),
        batch_size=128, shuffle=False, drop_last=False
    )
    perms = []

    test_acc_mat = np.zeros(shape=[n_subtask, n_subtask])
    test_loss_mat = np.zeros_like(test_acc_mat)
    for i in range(n_subtask):
        perm = net.change_permutation()
        perms.append(perm)

        # train on subtask i
        p = reunn.SupervisedClassificationTaskPipeline(
            net=net, log_dir="../log_dir/pmnist", backend="torch",
            hparam=None, device=device,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(net.parameters(), lr=1e-3),
            train_loader=train_loader, test_loader = test_loader
        )
        p.train(epochs=epochs, validation=False)

        # test on subtask 0~i
        for j, perm in enumerate(perms):
            if perm is not None:
                net.change_permutation(perm)
            result = p.test()
            test_loss_mat[i, j] = result["test_loss"]
            test_acc_mat[i, j] = result["test_acc"]
        print(
            f"<<<<<<<<<< {i+1} Learned Tasks, "
            f"mean_test_loss={np.mean(test_loss_mat[i, 0:i+1])}, ",
            f"mean_test_acc={np.mean(test_acc_mat[i, 0:i+1])}"
            f"<<<<<<<<<<"
        )

    f, _ = cltask.plot_permuted_benchmark_result_matrix(
        test_loss_mat=test_loss_mat, test_acc_mat=test_acc_mat
    )
    f.set_size_inches(12, 4)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="permuted_mnist")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.mode == "sequential_regression_overall":
        sequential_regression_overall_train(epochs=1500, device=args.device)
    elif args.mode in ("sequential_regression", "sequential_regression_phasic"):
        sequential_regression_phasic_train(epochs=3000, device=args.device)
    elif args.mode == "permuted_mnist":
        permuted_mnist(n_subtask=20, epochs=10, device=args.device)