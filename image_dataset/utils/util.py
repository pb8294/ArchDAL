import numpy as np
# from texttable import Texttable
import torch
import pandas as pd
import matplotlib.pyplot as plt
import math
import json

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def get_device(args):
    """

    Parameters
    ----------
    args : Argument parser object

    Returns
    -------
    device: cpu or gpu
    """
    return torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


# def table_printer(args):
#     """
#     Print the parameters of the model in a Tabular format
#     Parameters
#     ---------
#     args: argparser object
#         The parameters used for the model
#     """
#     args = vars(args)
#     keys = sorted(args.keys())
#     table = Texttable()
#     table.set_precision(4)
#     table.add_rows([["Parameter", "Value"]] +
#                    [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
#     print(table.draw())

def get_Z(model, sampler=1, reverse=False):
    """

    Parameters
    ----------
    model : trained neural network model

    Returns
    -------
    pi: activation level of each layer
    z: latent neural network architecture sampled from Beta-Bernoulli processes
    threshold: number of layers learnt
    """
    model_sampler = model.structure_sampler
    # model_sampler = model.encConv2.architecture_sampler
    if reverse:
        model_sampler = model.decConv1.architecture_sampler
    # if sampler == 2:
    #     model_sampler = model.structure_sampler_2
    # elif sampler == 3:
    #     model_sampler = model.structure_sampler_3
    # elif sampler == 4:
    #     model_sampler = model.structure_sampler_4
    Z, threshold, pi = model_sampler(get_pi=True, num_samples=sampler)
    z = Z.mean(0).cpu().detach().numpy()
    pi = pi.cpu().detach().numpy()
    return pi, z, threshold

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
    pi, z, threshold = get_Z(model, sampler=sampler, reverse=False)
    k_position = 50
    pi = pi.reshape(-1)
    # print(pi)
    pi[threshold:] = 0
    z = z[:, :threshold]
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
    cbar_ax = ax.scatter(x="x", y="y", c="data", s=sz, data=df, cmap='Blues', edgecolors="k", linewidths=0.2)
    ax.set_xlabel(r"Layers", fontsize=12)
    if ylabel:
        ax.set_ylabel(r"Active Neurons", fontsize=12)
        ax.yaxis.set_label_coords(0, 0.3)
    ax.invert_yaxis()
    
    return cbar_ax


def plot_prediction(ax, results, legend=False):
    """

    Parameters
    ----------
    ax : Matplotlib axis object
    results : dictionary that contains testing data points and predictions with standard deviation
    legend : Flag to add legend to the plot

    Returns
    -------

    """
    var = (results['total_unc'])
    ax.plot(results['xs_t'].ravel(), results['mean'], color='k', label="mean", linewidth=1, alpha=0.7)
    ax.fill_between(results['xs_t'].ravel(), results['mean'] + var, results['mean'] - var, color='green', alpha=0.3,
                    label="std")
    ax.scatter(results['x'].ravel(), results['y'].ravel(), s=1, c='gray', alpha=0.5, marker="o", label="data points")
    ax.set_ylim([-2.5, 3])
    ax.set_xlim([-2.5, 2.5])
    ax.set_xlabel("X", fontsize=12)
    ax.text(0.25, -0.2, '')

    ax.set_xticks(np.arange(-2, 3, 2))
    ax.set_xticklabels(np.arange(-2, 3, 2))
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    if legend:
        ax.scatter(results['x'].ravel(), results['y'].ravel(), s=5, c='gray', alpha=0.5, marker="o")
        ax.set_ylabel("Y", fontsize=12)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.2)
    return ax

# Check if indices in one list is available in another list
def check_uniq_indices(l1, l2):
    for l in l1:
        if l in l2:
            return True
    
    return False

def plot_network_mask_all(net, savefilepath):
    plt.figure(figsize=(20, 10), dpi=300)
    ax1 = plt.subplot(141)
    plot_network_mask(ax1, net, ylabel=True, sz=10, sampler=1)
    ax2 = plt.subplot(142)
    plot_network_mask(ax2, net, ylabel=True, sz=10, sampler=2)
    ax3 = plt.subplot(143)
    plot_network_mask(ax3, net, ylabel=True, sz=10, sampler=3)
    ax4 = plt.subplot(144)
    plot_network_mask(ax4, net, ylabel=True, sz=10, sampler=4)
    plt.savefig(savefilepath)
    
def plot_network_mask_base(net, savefilepath, figsize=False, reverse=False):
    if figsize:
        plt.figure()
    else:
        plt.figure(figsize=(10, 30))
    ax = plt.gca()
    plot_network_mask(ax, net, ylabel=True, sz=7, sampler=15, reverse=False)
    plt.tight_layout()
    plt.savefig(savefilepath)
    
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    


#  function  = 1 − cos(a), where a scans from 0 to pi/2

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L 

def geometric_discount(n, i):
    return 2 ** (n - i) / (2 ** n - 1)

# Added by Pradeep Bajracharya
def get_indices(dataset,class_name, datatype="MNIST"):
    np.random.seed(1)
    if datatype != "SVHN":
        labels = dataset.targets
    else:
        labels = dataset.labels

    indices =  []
    for i in range(len(labels)):
        if labels[i] == class_name:
            indices.append(i)
    np.random.shuffle(indices)
    return indices