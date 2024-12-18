import sys
# from tkinter import NW

# from torchinfo import summary
from utils.models.MLP import AdaptiveMLP, simpleMLP
sys.path.append('utils/')
sys.path.append('distil/')

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
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms



# Distil libraries for model and training
from utils.models.CNN import AdaptiveResnet, AdaptiveConvNet, AdaptiveRN, simpleCNN
from distil.utils.models.resnet import ResNet18, ResNet34, ResNet50
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

aug_st = "aug"
if args.aug == 0:
    aug_st = "n_aug"
dataset_root_path = 'path_to_data/datahub'
save_dirname = str(args.network) + "-" + str(args.dataset) + "-" + str(args.strategy) + "-" + str(budget) + "-" + str(aug_st)

if not os.path.exists("results"):
    os.mkdir("results")
    
if not os.path.exists("results/" + str(args.exp)):
    os.mkdir("results/" + str(args.exp))
             
if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname)
    
if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run)

if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run + "/" + args.seed):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run + "/" + args.seed)

save_loc = "results/" + str(args.exp) + "/" + save_dirname + "/" + args.run + "/" + args.seed
model_directory = save_loc + "/m_latest.pt"

if not os.path.isdir(save_loc):
    os.makedirs(save_loc)

# Train on approximately the full dataset given the budget contraints
n_rounds = (training_size_cap - initial_seed_size) // budget

print(save_loc)
print(model_directory)

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

args.nclasses = nclasses
args.nchannel = nchannel

print(args)
dim = full_train_dataset[0][0].shape

# Commented to create a validation size for model save
# initial_train_indices = np.random.choice(len(full_train_dataset), replace=False, size=initial_seed_size)

# Old Version Matching
num_train = len(full_train_dataset)
indices = np.arange(len(full_train_dataset))
print(num_train, indices)
initial_train_indices = []
for i in range(10):
    class_ind = get_indices(full_train_dataset, i, args.dataset)
    np.random.shuffle(class_ind)
    initial_train_indices.append(class_ind[:100])

initial_train_indices = np.array(initial_train_indices).flatten()
np.random.shuffle(initial_train_indices)

# Get all indices except initially labelled set
non_init_idx = indices[~np.isin(indices, initial_train_indices)]
np.random.shuffle(non_init_idx)

# Split the data into train and valid set (90 - 10 split)
split = int(np.floor(args.valid_size * num_train))
full_train_indices, valid_train_indices = non_init_idx[split:], non_init_idx[:split]
np.random.shuffle(full_train_indices)
full_train_indices = full_train_indices[:30000]
print(len(np.unique(full_train_indices)))

init_subset = Subset(full_train_dataset, initial_train_indices)
pool_subset = Subset(full_train_dataset, full_train_indices)
valid_subset = Subset(full_train_dataset, valid_train_indices)
valid_loader = DataLoader(valid_subset, shuffle=True, batch_size=64, num_workers=4)
train_loader = DataLoader(init_subset, shuffle=True, batch_size=64, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)
pool_loader = DataLoader(pool_subset, shuffle=False, batch_size=1, num_workers=4)

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
args = vars(args)
# print(net)
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print(pytorch_total_params)
net = net.to(device)

load_model = False if args["load"] == 0 else True

st_time = time.time()
if load_model:
    net.load_state_dict(torch.load(model_directory))
    initial_model = net
else:
    dt = data_train(Subset(full_train_dataset, initial_train_indices), net, args)
    if nw is not None:
        plot_network_mask_base(net, save_loc + "/model_before_init.png", figsize=figbool)#; exit(0)
    initial_model, _ = dt.train(None, valid_loader=valid_loader, test_loader=test_loader, save_loc=save_loc, save_path=model_directory, nw=nw)
    
if nw == 'ada1':
    plot_network_mask_all(net, save_loc + "/model_after_init200.png")
    plot_network_mask_base(net, save_loc + "/model_A_init200.png", figsize=figbool)

print("Training for", n_rounds, "rounds with budget", budget, "on unlabeled set size", training_size_cap, " -- Time taken: ", str(time.time() - st_time))
# exit(0)

initial_model.load_state_dict(torch.load(model_directory))
strat_logs = save_loc
acc, loss = train_one(full_train_dataset, initial_train_indices, valid_train_indices, test_loader, copy.deepcopy(initial_model), n_rounds, budget, args, nclasses, args["strategy"], strat_logs, save_loc, "model_{}.pt".format(args["strategy"]), nw=nw)
with open(os.path.join(save_loc,'acc.txt'), 'w') as f:
    for item in acc:
        f.write("%s\n" % item)
        
with open(os.path.join(save_loc,'loss.txt'), 'w') as f:
    for item in loss:
        f.write("%s\n" % item)

