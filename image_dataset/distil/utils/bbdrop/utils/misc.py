import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def accuracy(outs, targets):
    _, preds = torch.max(outs.data, -1)
    return float(preds.eq(targets).sum().item())/outs.size(0)

def freeze_batch_norm(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.eval()

def add_args(args1, args2):
    for k, v in args2.__dict__.items():
        args1.__dict__[k] = v

def plot_network_mask(ax, model, ylabel=False, sz=7, sampler=1, reverse=False):
    """

    Parameters
    ----------
    ax : Matplotlib axis object
    model : Trained neural network model
    ylabel : Flag to add y-axis label

    Returns
    -------
    cbar_ax: matplotlib axes to add colorbar

    """
    z = torch.empty(model.n_layers, model.filters)
    counts = torch.empty(model.n_layers)
    for ii, m in enumerate(model.gated_layers):
        z[ii] = m.get_mask().detach().cpu()
        counts[ii] = m.get_num_active()
    z = z.numpy()
    counts = counts.numpy()
    k_position = 50
    pi = counts / model.filters
    # print(pi)
    scale = 15
    pi = pi * scale
    x_data = np.arange(z.shape[1])
    XX, YY = np.meshgrid(x_data, np.arange(z.shape[0]))
    table = np.vstack((XX.ravel(), YY.ravel(), z.ravel())).T
    
    df = pd.DataFrame(table)
    df.columns = ['x', 'y', 'data']

    ax.set(frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(-4.8, -(scale / 2), r"$\pi$", fontsize=16)
    ax.text(k_position/2, -2.3, r"$K^+$", fontsize=16)
    ax.text(-4.5, -0.3, r"0", fontsize=14)
    ax.text(-4.5, -1 * scale - 0.5, r"1", fontsize=14)
    ax.hlines(y=-scale - 0.5, xmin=-0.2, xmax=len(pi), linewidth=1, color="k", linestyle="-")
    ax.bar(np.arange(len(pi)), -1. * pi, bottom=[-1] * len(pi), color="black", width=0.7)
    ax.hlines(y=-0.5, xmin=-0.2, xmax=len(pi), linewidth=1, color="k")
    ax.set_xlim(-5, k_position-1)
    cbar_ax = ax.scatter(x="y", y="x", c="data", s=sz, data=df, cmap='Blues', edgecolors="k", linewidths=0.2)
    ax.set_xlabel(r"Layers", fontsize=12)
    if ylabel:
        ax.set_ylabel(r"Active Neurons", fontsize=12)
        ax.yaxis.set_label_coords(0, 0.3)
    ax.invert_yaxis()
    
    return cbar_ax

def plot_network_mask_base(net, savefilepath, figsize=False, reverse=False):
    if figsize:
        plt.figure()
    else:
        plt.figure(figsize=(5, 5))
    ax = plt.gca()
    plot_network_mask(ax, net, ylabel=True, sz=7, sampler=15, reverse=False)
    plt.tight_layout()
    plt.savefig(savefilepath)