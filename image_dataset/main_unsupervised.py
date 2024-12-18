import sys
from tkinter import NW

from torchinfo import summary
# from distil.utils.utils import LabeledToUnlabeledIdxDataset
from utils.models.MLP import AdaptiveMLP, simpleMLP
sys.path.append('utils/')
sys.path.append('distil/')

import faiss

# Data processing libraries
import pandas as pd 
import numpy as np
from numpy.linalg import cond
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls


from backpack import backpack
from backpack.extensions import (BatchGrad, BatchL2Grad)
from backpack import extend
import torch.nn as nn


# Miscellaneous Libraries
import time
import math
import random
import os
import pickle
import argparse

# Image libraries
import copy
from PIL import Image
import matplotlib.pyplot as plt

# Torch Libraries
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

# from distil.active_learning_strategies.ntk import NTK

# Distil libraries for model and training
from utils.models.CNN import AdaptiveResnet, AdaptiveConvNet, AdaptiveRN, simpleCNN
from distil.utils.models.resnet import ResNet18, ResNet34, ResNet50
# from distil.utils.models.ntk import VGG_11, ResNet, ResNet_18, ResNet_34, ResNet_50
from utils.models.vae import vae_conv_bottleneck
from distil.utils.models.vgg import VGG
from distil.utils.train_helper import data_train

# utils library for training
from utils.training import train_one
from utils.util import plot_network_mask_all, plot_network_mask_base, get_indices

import warnings
warnings.filterwarnings("ignore")

# args = {'n_epoch':300, 'lr':float(0.01), 'batch_size':20, 'max_accuracy':float(0.99), 'islogs':True, 'isreset':True, 'isverbose':True, 'device':'cuda'} 

def argument_parser():
    parser = argparse.ArgumentParser(description="Active learning experiments to study effect of network architecture")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the randomness")
    parser.add_argument('--exp', type=str, default="base_ada_convnet_june_100",
                        help='Directory for results to store')
    parser.add_argument('--run', type=str, default="run1",
                        help='Sub directory for results to store')
    parser.add_argument('--cuda', action='store_false', 
                        help="Use GPU or CPU")
    
    # Dataset and network
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN")
    parser.add_argument("--valid_size", type=float, default=0.1,
                        help="Percentage of full train dataset to consider as valid set")
    parser.add_argument("--network", type=str, default="resnet",
                        help="resnet, vgg, cnn, mlp")
    parser.add_argument('--load', type=int, default=0,
                        help="Load initialized network")
    parser.add_argument('--aug', type=int, default=0,
                        help="Augment Data")
    
    # Training configs
    parser.add_argument("--n_epoch", type=int, default=500,                
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-6,
                        help="Coefficient of weight decay.")
    
    # 20 for CIFAR and MNIST, 64 for SVHN FMNIST?
    parser.add_argument("--batch_size", type=float, default=64,                 
                        help="Batch size.")
    parser.add_argument("--strategy", type=str, default="random",
                        help="random, entropy, confidence, badge, coreset, margin, bald")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        help="sgd, adam")
    parser.add_argument("--es", type=int, default=0,
                        help="0 or 1")
    
    # Verbose configs
    parser.add_argument('--islogs', action='store_false', 
                        help="Log results")
    parser.add_argument('--isreset', action='store_false', 
                        help="Reset network")
    parser.add_argument('--isverbose', action='store_false', 
                        help="Show verbose results")
    
    parser.add_argument("--prior_temp", type=float, default=1.,
                        help="Temperature for Concrete Bernoulli from prior")
    parser.add_argument("--temp", type=float, default=.5,                 
                        help="Temperature for Concrete Bernoulli from posterior")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Epsilon to select the activated layers")
    parser.add_argument("--truncation_level", type=int, default=20,
                        help="K+: Truncation for Z matrix")
    parser.add_argument("--a_prior", type=float, default=1,
                        help="a parameter for Beta distribution")
    parser.add_argument("--b_prior", type=float, default=10.,
                        help="b parameter for Beta distribution")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples of Z matrix")
    parser.add_argument("--max_width", type=int, default=64,
                        help="Dimension of hidden representation.")
    parser.add_argument("--budget", type=int, default=1000,
                        help="Acquisition size budget")
    parser.add_argument("--pretrain", type=int, default=1,
                        help="Unsupervised-pretrain")
    return parser.parse_known_args()[0]

args = argument_parser()
if args.cuda:
    args.device = "cuda"
else:
    args.device = "cpu"
    
if args.dataset == "SVHN" or args.dataset == 'FashionMNIST':
    args.batch_size = 64
# print(args); exit(0)
# Set seed and setup for reproducible result everytime
print(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

initial_seed_size = 1000    # INIT SEED SIZE HERE
training_size_cap = 31000   # TRAIN SIZE CAP HERE
budget = args.budget               # BUDGET HERE

# fn = ["aug", "n_aug"]
aug_st = "aug"
if args.aug == 0:
    aug_st = "n_aug"
dataset_root_path = '/home/pb8294/Documents/Projects/ALArch/datahub'
save_dirname = str(args.network) + "-" + str(args.dataset) + "-" + str(args.strategy) + "-" + str(budget) + "-" + str(aug_st)

if not os.path.exists("results"):
    os.mkdir("results")
    
if not os.path.exists("results/" + str(args.exp)):
    os.mkdir("results/" + str(args.exp))
             
if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname)
    
if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run)

# save_loc = "results/" + str(args.exp) + "/" + save_dirname
save_loc = "results/" + str(args.exp) + "/" + save_dirname + "/" + args.run
model_directory = save_loc + "/m_latest.pt"

if not os.path.isdir(save_loc):
    os.makedirs(save_loc)

# Train on approximately the full dataset given the budget contraints
n_rounds = (training_size_cap - initial_seed_size) // budget

print(save_loc)
print(model_directory)
#!/content/gdown.pl/gdown.pl "INSERT SHARABLE LINK HERE" "INSERT DOWNLOAD LOCATION HERE (ideally, same as model_directory)" # MAY NOT NEED THIS LINE IF NOT CLONING MODEL FROM COLAB

# Load the dataset

mlp_dim = 1024
if args.dataset == "CIFAR10":
    if args.aug == 1:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    full_train_dataset = datasets.CIFAR10(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.CIFAR10(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

    nclasses = 10
    nchannel = 3
    mlp_dim = 1024 * 3
    image_dim=32

elif args.dataset == "CIFAR100":
    if args.aug == 1:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    full_train_dataset = datasets.CIFAR100(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.CIFAR100(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

    nclasses = 100
    nchannel = 3
    mlp_dim = 1024 * 3

elif args.dataset == "MNIST":
    image_dim=28
    if args.network == "vgg11" or args.network == "vgg16" or args.network == "vgg19":
        image_dim=32
    if args.aug == 1:
        train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.RandomCrop(image_dim, padding=4), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    full_train_dataset = datasets.MNIST(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.MNIST(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

    nclasses = 10
    nchannel = 1
    mlp_dim = 784

elif args.dataset == "FashionMNIST":
    image_dim=28
    if args.network == "vgg11" or args.network == "vgg16" or args.network == "vgg19":
        image_dim=32
    if args.aug == 1:
        train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.RandomCrop(image_dim, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
       train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Use mean/std of MNIST 

    full_train_dataset = datasets.FashionMNIST(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.FashionMNIST(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

    nclasses = 10
    nchannel = 1
    mlp_dim = 784

elif args.dataset == "SVHN":

    if args.aug == 1:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

    full_train_dataset = datasets.SVHN(dataset_root_path, split='train', download=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.SVHN(dataset_root_path, split='test', download=True, transform=test_transform, target_transform=torch.tensor)

    nclasses = 10
    nchannel = 3
    mlp_dim = 1024 * 3
    image_dim=32
# exit(0)
# args['nclasses'] = nclasses
args.nclasses = nclasses
args.nchannel = nchannel
# args.optimizer='sgd'
# args = vars(args)
print(args)
# exit(0)
dim = full_train_dataset[0][0].shape

# Commented to create a validation size for model save
# initial_train_indices = np.random.choice(len(full_train_dataset), replace=False, size=initial_seed_size)

# Old Version Matching
num_train = len(full_train_dataset)
indices = np.arange(len(full_train_dataset))
# print(num_train, indices)
initial_train_indices = []
for i in range(10):
    class_ind = get_indices(full_train_dataset, i, args.dataset)
    np.random.shuffle(class_ind)
    initial_train_indices.append(class_ind[:100])

initial_train_indices = np.array(initial_train_indices).flatten()
# print(np.sort(initial_train_indices))
# exit(0)
np.random.shuffle(initial_train_indices)

# Get all indices except initially labelled set
non_init_idx = indices[~np.isin(indices, initial_train_indices)]
np.random.shuffle(non_init_idx)
# exit(0)

# Split the data into train and valid set (90 - 10 split)
split = int(np.floor(args.valid_size * num_train))
full_train_indices, valid_train_indices = non_init_idx[split:], non_init_idx[:split]
# pool_idx = np.random.choice(pool_idx, 30000)
np.random.shuffle(full_train_indices)
# remains = full_train_indices[30000:]
full_train_indices = full_train_indices[:1000]
print(len(np.unique(full_train_indices)))
# Dataloader for the initial labelled set, test set and validation set
# print(initial_train_indices, len(initial_train_indices))
# print(initial_train_indices)
init_subset = Subset(full_train_dataset, initial_train_indices)
pool_subset = Subset(full_train_dataset, full_train_indices)



valid_subset = Subset(full_train_dataset, valid_train_indices)
valid_loader = DataLoader(valid_subset, shuffle=True, batch_size=64, num_workers=4)
train_loader = DataLoader(init_subset, shuffle=True, batch_size=64, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)
pool_loader = DataLoader(pool_subset, shuffle=False, batch_size=1, num_workers=4)





# Updated by Pradeep Bajracharya
# full_train_indices = np.arange(len(full_train_dataset))
# initial_train_indices = np.random.choice(full_train_indices, replace=False, size=initial_seed_size)
# full_train_indices = np.delete(full_train_indices, initial_train_indices)
# valid_train_indices = np.random.choice(full_train_indices, replace=False, size=int(len(full_train_dataset) * args.valid_size))

valid_set = Subset(full_train_dataset, valid_train_indices)
print(len(valid_loader.dataset), len(train_loader.dataset), len(test_loader.dataset))
print("Initially --> Valid loader size: {}, labelled size: {}, test size: {}, pool size: {}".format(len(valid_train_indices), len(initial_train_indices), len(test_dataset), len(full_train_indices)))
print(full_train_indices[-10:])
print(initial_train_indices[-10:])
print(valid_train_indices[-10:])
# exit(0)

# Define the network and determine to load or intialize the network
device = "cuda" if torch.cuda.is_available() else "cpu"
print(args.network)
nw = None
figbool=False

def preprocess_features(npdata, pca=128):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Using PCA didn't help in our case.
    
    # Apply PCA-whitening with Faiss
    #mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.9)
    #mat.train(npdata)
    #assert mat.is_trained
    #npdata = mat.apply_py(npdata)


    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 10
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


def _train(epoch, loader_tr, optimizer, net, device, nw=None, weight=1, opt_for=None, valid_set=None,):
        net.train()
        accFinal = 0.
        # Pradeep Bajracharya
        lossFinal = 0.
        criterion = torch.nn.CrossEntropyLoss()
        criterion.reduction = "mean"
        klFinal = 0.
        klScale = 0.
        eLog = 0.
        accs_arr = []
        train_accs = 0.
        # print("Epoch: ", epoch, "Nw:", nw, "Opt for:", opt_for)
        for batch_id, (x, y) in enumerate(loader_tr):
            
            x, y = x.float().to(device=device), y.to(device=device)
            # y = y.squeeze(1)
            
            optimizer.zero_grad()
            # print(x.shape)
            out = net(x)
            # out = out.squeeze(0)
            # print(out.shape, y.shape)
            # exit(0)
            # print(out.shape, y.shape)
            # print(out[-1], y[-1])
            if nw is None:
                # print("hh")
                loss = criterion(out, y.long())
                kls, kl_scale, e_total = 0.0, 0.0, 0.0
            else:
                loss, kls, kl_scale, e_total= net.estimate_ELBO(criterion, out, y, len(loader_tr.dataset), kl_weight=weight)

            if nw == 'ada1':
                out = out.mean(0)
            loss.backward()

            optimizer.step()
            with torch.no_grad():
                y_pred = torch.max(out,1)[1]
                
                accFinal += torch.sum(y_pred == y).float().item()
            lossFinal += loss.item()
            if nw is not None:
                klFinal += kls.item()
                klScale += kl_scale
                eLog += e_total.item()
        return accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr), klFinal / len(loader_tr), klScale / len(loader_tr), eLog / len(loader_tr)
    
from sklearn.metrics import accuracy_score
def get_acc_on_set(net, loader_te, device, nw=None):
        
        """
        Calculates and returns the accuracy on the given dataset to test
        
        Parameters
        ----------
        test_dataset: torch.utils.data.Dataset
            The dataset to test
        Returns
        -------
        accFinal: float
            The fraction of data points whose predictions by the current model match their targets
        """	


        if loader_te is None:
            raise ValueError("Test loader not present")
        
        net.eval()
        accFinal = 0.
        
        # Pradeep Bajracharya
        lossFinal = 0.
        loglike = 0.
        criterion = torch.nn.CrossEntropyLoss()
        criterion.reduction = "mean"
        
        y_actual = None
        y_predic = None		

        with torch.no_grad():        
            # self.clf = self.clf.to(device=self.device)
            for batch_id, (x,y) in enumerate(loader_te):     
                x, y = x.float().to(device=device), y.to(device=device)
                # print(y[-20:-15]); exit(0)

                if nw == "ada1":
                    out = net(x, 2)
                else:
                    out = net(x)
                # print(out[-1]); exit()
                # Pradeep Bajracharya
                if nw == "ada1":
                    logits = F.softmax(out.mean(0), dim=1)
                else:
                    logits = F.softmax(out, dim=1)
                ll = -F.nll_loss(logits, y, reduction="sum").item()
                loglike += ll

                if nw == "ada1":
                    out = out.mean(0)
                # loss = criterion(out, y.long())
                lossFinal += F.cross_entropy(logits, y, reduction="mean").item()
                y_pred = torch.max(out,1)[1].detach().cpu().numpy().flatten()
                predicted = torch.argmax(logits, 1)
                
                if y_predic is None:
                    y_predic=y_pred
                    y_actual=y.detach().cpu().numpy().flatten()
                else:
                    y_predic=np.hstack((y_predic,y_pred))
                    y_actual=np.hstack((y_actual,y.detach().cpu().numpy().flatten()))
                
                # accFinal += torch.sum(1.0*(y_pred == y)).item() #.data.item()
        
        
        acc_final = accuracy_score(y_actual, y_predic)
        
        # Pradeep Bajracharya
        return acc_final, lossFinal / len(loader_te)
# exit()
mlt = 5
if args.pretrain == 100:
    all_train_indices = indices[~np.isin(indices, valid_train_indices)]
    all_train_set = Subset(full_train_dataset, all_train_indices)
    all_loader = DataLoader(all_train_set, shuffle=False, batch_size=64, num_workers=4)

    if args.network == 'resnet18':
        net = ResNet18(num_classes=10 * mlt, channels=args.nchannel)
    elif args.network == 'resnet34':
        net = ResNet34(num_classes=10 * mlt, channels=args.nchannel)
    elif args.network == 'resnet50':
        net = ResNet50(num_classes=10 * mlt, channels=args.nchannel)
    elif args.network == 'resnetada':
        args.max_width = 64
        nw = "ada1"
        net = AdaptiveRN(args.nchannel, num_classes=nclasses, args=args).to(device)

    elif args.network == 'vgg11':
        net = VGG('VGG11', num_classes=10 * mlt, in_channels=args.nchannel)
    elif args.network == 'vgg16':
        net = VGG('VGG16', num_classes=10 * mlt, in_channels=args.nchannel)
    elif args.network == 'vgg19':
        net = VGG('VGG19', num_classes=10 * mlt, in_channels=args.nchannel)


    elif args.network == "adamlp":
        nw = "ada1"
        figbool = False
        net = AdaptiveMLP(input_feature_dim=mlp_dim, out_feature_dim=nclasses * mlt, args=args, device=device).to(device)
    elif args.network == "adacnn":
        nw = "ada1"
        figbool = True
        net = AdaptiveConvNet(input_channels=args.nchannel, num_classes=nclasses, args=args, device=device).to(device)
    elif args.network == "simpleMLP":
        nw = None
        net = simpleMLP(input_feature_dim=mlp_dim, out_feature_dim=nclasses * mlt, args=args, device=device).to(device)
    elif args.network == "simpleCNN":
        nw = None
        net = simpleCNN(input_channels=args.nchannel, num_classes=nclasses * mlt, args=args, device=device).to(device)

    elif args.network == "AdaVAE":
        nw = "ada1"
        net = vae_conv_bottleneck(input_channel=args.nchannel, hidden_channel=64, truncation=args.truncation_level, device=device).to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                        momentum=0.9,
                        weight_decay=args.l2,
                        nesterov=True)

    net = net.to(device)     
    best_acc = -np.Inf
    patience = 30
    early_stop = 0
    save_every = 1
    
    print("total epochs: {}".format(args.n_epoch))
    print(net)
    for epoch in range(args.n_epoch):
        print("\n Current epoch: {}".format(epoch))
        # input_all, embedding_all = [], []
        # for batch_idx, (x, y) in enumerate(all_loader):
        #     x, y = x.float().to(device=device), y.to(device=device)
        #     out, e = net(x, last=True)
        #     # print(e.shape)
        #     embedding_all.append(e.detach().cpu().squeeze(0))
        #     input_all.append(x.detach().cpu())
            
        # print("All embedding found")
        # embedding_all = torch.cat(embedding_all)
        # embedding_all = np.asarray(embedding_all.numpy())
        # input_all = np.asarray(torch.cat(input_all).numpy())
        # print(embedding_all.shape, type(embedding_all), input_all.shape)
        # # exit(0)
        # print("Running KMeans. Getting Pseudo label")
        # processed_e = preprocess_features(embedding_all)
        # print(processed_e.shape)
        # # exit(0)
        # km_I, _ = run_kmeans(processed_e, 10 * mlt)
        # # print(km_I)
        # input_all = torch.tensor(input_all)
        # km_I = torch.tensor(km_I)
        # print(km_I)
        # # print(type(input_all), type(km_I), input_all.shape, km_I.shape)
        # my_dataset = torch.utils.data.TensorDataset(input_all, km_I)
        # all_loader_pretrain = DataLoader(my_dataset, shuffle=True, batch_size=64)
        
        t_start = time.time()
        accCurrent, lossCurrent, _, _, _ = _train(epoch, train_loader, optimizer, net, device, nw=nw)
        t_end = time.time()
        

        print("{}, {}, Training time: {}".format(accCurrent, lossCurrent, t_end - t_start))
        print(net.structure_sampler.get_variational_params())
        if epoch % save_every == 0 and (epoch > 0):
            stored_model_keys = list(net.state_dict().keys())[:-2]
            msd =  net.state_dict()
            pretrained_dict = {key:msd[key] for key in stored_model_keys}
            with open('%s/pretrained_%s_%s_lr_%s_batch_%s_%s_final.pickle' % 
                      (save_loc, args.dataset, args.network, args.lr, args.batch_size, args.n_epoch), 'wb') as handle:
                pickle.dump(pretrained_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
            
        
    exit(0)


    

if args.network == 'resnet18':
    net = ResNet18(channels=args.nchannel)
elif args.network == 'resnet34':
    net = ResNet34(channels=args.nchannel)
elif args.network == 'resnet50':
    net = ResNet50(channels=args.nchannel)
elif args.network == 'vgg11':
    net = VGG('VGG11', in_channels=args.nchannel)
elif args.network == 'vgg16':
    net = VGG('VGG16', in_channels=args.nchannel)
elif args.network == 'vgg19':
    net = VGG('VGG19', in_channels=args.nchannel)
elif args.network == "adacnn":
    nw = "ada1"
    figbool = True
    net = AdaptiveConvNet(input_channels=args.nchannel, num_classes=nclasses, args=args, device=device).to(device)
elif args.network == "simpleCNN":
    nw = None
    net = simpleCNN(input_channels=args.nchannel, num_classes=nclasses, args=args, device=device).to(device)



pretrain_loc = "results/ada_pt_unsupervised/" + str(args.network) + "-" + str(args.dataset) + "-random-" + str(budget) + "-" + str(aug_st) + "/" + args.run
args.pretrain_path = pretrain_loc
if args.pretrain == 1:
    pretraining_path = '%s/pretrained_%s_%s_lr_%s_batch_%s_%s_final.pickle' % (pretrain_loc, args.dataset, args.network, args.lr, args.batch_size, 500)
    with open(pretraining_path, "rb") as f:
        pretrain_w = pickle.load(f)
    model_dict = net.state_dict()
    model_dict.update(pretrain_w)
    net.load_state_dict(model_dict)
    print("loaded pretrained dictionary")
args = vars(args)
print(net)
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
net = net.to(device)
net = extend(net)
load_model = False if args["load"] == 0 else True

summary(net, input_size=(1, nchannel, image_dim, image_dim))

st_time = time.time()
if load_model:
    net.load_state_dict(torch.load(model_directory))
    initial_model = net
else:
    dt = data_train(Subset(full_train_dataset, initial_train_indices), net, args)
    if nw is not None:
        plot_network_mask_base(net, save_loc + "/model_before_init.png", figsize=figbool)#; exit(0)
    print('here', args["n_epoch"])
    # exit(0)
    initial_model, _ = dt.train(None, valid_loader=valid_loader, test_loader=test_loader, save_loc=save_loc, save_path=model_directory, nw=nw)
    
if nw == 'ada1':
    plot_network_mask_all(net, save_loc + "/model_after_init200.png")
    plot_network_mask_base(net, save_loc + "/model_A_init200.png", figsize=figbool)
    # torch.save(initial_model.state_dict(), model_directory)


print("Training for", n_rounds, "rounds with budget", budget, "on unlabeled set size", training_size_cap, " -- Time taken: ", str(time.time() - st_time))
# exit(0)


initial_model.load_state_dict(torch.load(model_directory))
strat_logs = save_loc
# strategy = NTK(train_dataset, LabeledToUnlabeledIdxDataset(pool_subset), net, nclasses, strategy_args)
# idx = strategy.select(budget) 
# exit(0)
acc, loss = train_one(full_train_dataset, initial_train_indices, valid_train_indices, test_loader, copy.deepcopy(initial_model), n_rounds, budget, args, nclasses, args["strategy"], strat_logs, save_loc, "model_{}.pt".format(args["strategy"]), nw=nw)
with open(os.path.join(save_loc,'acc.txt'), 'w') as f:
    for item in acc:
        f.write("%s\n" % item)
        
with open(os.path.join(save_loc,'loss.txt'), 'w') as f:
    for item in loss:
        f.write("%s\n" % item)

