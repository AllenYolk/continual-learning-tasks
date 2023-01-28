import torch
import torch.nn as nn
from torch import optim

import cltask
import reunn


def sequential_regression_overall_train(epochs: int):
    net = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    st = reunn.NetStats(net=net, input_shape=[1, 1])
    st.print_summary()

    def f_target(x):
        return torch.sin(x*2+0.3) + 0.5*torch.sin(x*3-0.2)

    ranges = [[x, x+2] for x in range(-5, 5, 2)]
    _, test_loader = cltask.get_sequential_regression_loader(
        fx=f_target, ranges=ranges, batch_size=100
    )

    cltask.plot_sequential_regression(
        model=net, fx=f_target, start=-5, end=5
    )
    p = reunn.SupervisedTaskPipeline(
            backend="torch", log_dir="../log_dir", net=net, 
            criterion=nn.MSELoss(), 
            optimizer=optim.Adam(net.parameters(), lr=1e-3),
            train_loader=test_loader,
            validation_loader=test_loader
        )
    p.train(epochs=epochs, validation=True, silent=True)
    cltask.plot_sequential_regression(
        model=net, fx=f_target, start=-5, end=5
    )


def sequential_regression_phasic_train(epochs: int):
    net = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    st = reunn.NetStats(net=net, input_shape=[1, 1])
    st.print_summary()

    def f_target(x):
        return torch.sin(x*2+0.3) + 0.5*torch.sin(x*3-0.2)

    ranges = [[x, x+2] for x in range(-5, 5, 2)]
    train_loaders, test_loader = cltask.get_sequential_regression_loader(
        fx=f_target, ranges=ranges, batch_size=100
    )

    phases = len(ranges)
    cltask.plot_sequential_regression(
        model=net, fx=f_target, start=-5, end=5, #title="original"
        fill_x_ranges=ranges
    )
    for phase in range(phases):
        p = reunn.SupervisedTaskPipeline(
            backend="torch", log_dir="../log_dir", net=net, 
            criterion=nn.MSELoss(), 
            optimizer=optim.Adam(net.parameters(), lr=1e-3),
            train_loader=train_loaders[phase],
            validation_loader=test_loader
        )
        p.train(epochs=epochs, validation=True, silent=True)
        cltask.plot_sequential_regression(
            model=net, fx=f_target, start=-5, end=5, dx=0.02,
            title=f"phase {phase}: {ranges[phase]}",
            fill_x_ranges=[ranges[phase]]
        )


if __name__ == "__main__":
    #sequential_regression_overall_train(epochs=1500)
    sequential_regression_phasic_train(epochs=3000)