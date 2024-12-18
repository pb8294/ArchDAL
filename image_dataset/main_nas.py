import os
import pdb
import shutil
import sys
import time
import glob
import numpy as np
from distil.utils.train_helper import data_train
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from utils.models.model_search import Network
from utils.models.genotypes import PRIMITIVES, Genotype
from torchvision import datasets, transforms
from utils.visualize import plot
from scipy import stats

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def init_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    #gram = np.matmul(X[indsAll], X[indsAll].T)
    #val, _ = np.linalg.eig(gram)
    #val = np.abs(val)
    #vgt = val[val > 1e-2]
    return indsAll

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

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

# Train Config
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--seed', type=int, default=2, help='random seed')


# Network and dataset
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--dataset', type=str, default="mnist", help='dataset: mnist/fashionmnist/svhn/cifar10')
parser.add_argument('--aug', type=int, default=0,
                        help="Augment Data")
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')

# Save configs
parser.add_argument("--budget", type=int, default=1000,
                        help="Acquisition size budget")
parser.add_argument('--strategy', type=str, default='random', help='Acqusition functions')
parser.add_argument('--exp', type=str, default="base_ada_convnet_june_100",
                        help='Directory for results to store')
parser.add_argument('--run', type=str, default="run1",
                    help='Sub directory for results to store')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
args = parser.parse_args()

initial_seed_size = 1000    # INIT SEED SIZE HERE
training_size_cap = 31000   # TRAIN SIZE CAP HERE
budget = args.budget               # BUDGET HERE

aug_st = "aug"
if args.aug == 0:
    aug_st = "n_aug"
dataset_root_path = 'path_to_data/datahub'
save_dirname = str("pdarts") + "-" + str(args.dataset) + "-" + str(args.strategy) + "-" + str(budget) + "-" + str(aug_st)

if not os.path.exists("results"):
    os.mkdir("results")
    
if not os.path.exists("results/" + str(args.exp)):
    os.mkdir("results/" + str(args.exp))
             
if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname)
    
if not os.path.exists("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run + "/" + args.seed):
    os.mkdir("results/" + str(args.exp) + "/" + save_dirname + "/" + args.run + "/" + args.seed)

# save_loc = "results/" + str(args.exp) + "/" + save_dirname
save_loc = "results/" + str(args.exp) + "/" + save_dirname + "/" + args.run + "/" + args.seed
model_directory = save_loc + "/m_latest.pt"

if not os.path.isdir(save_loc):
    os.makedirs(save_loc)

# args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
args.save = save_loc

create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == "cifar100":
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
elif args.dataset == "cifar10":
    input_channel=3
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'
elif args.dataset == "mnist":
    input_channel=1
    CIFAR_CLASSES = 10
    data_folder = 'mnist-10-batches-py'
elif args.dataset == "fashionmnist":
    input_channel=1
    CIFAR_CLASSES = 10
    data_folder = 'fashionmnist-10-batches-py'
elif args.dataset == "svhn":
    CIFAR_CLASSES = 10
    input_channel=3
    data_folder = 'svhn-10-batches-py' 
args.input_channel = input_channel
args.CIFAR_CLASSES = CIFAR_CLASSES 
if args.dataset == "cifar10":
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    full_train_dataset = datasets.CIFAR10(data_folder, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.CIFAR10(data_folder, download=True, train=False, transform=test_transform, target_transform=torch.tensor)
elif args.dataset == "mnist":
    image_dim = 32
    train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train_dataset = datasets.MNIST(data_folder, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.MNIST(data_folder, download=True, train=False, transform=test_transform, target_transform=torch.tensor)
elif args.dataset == "fashionmnist":
    image_dim = 32
    train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Use mean/std of MNIST 

    full_train_dataset = datasets.FashionMNIST(data_folder, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.FashionMNIST(data_folder, download=True, train=False, transform=test_transform, target_transform=torch.tensor)
elif args.dataset == "svhn":
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

    full_train_dataset = datasets.SVHN(data_folder, split='train', download=True, transform=train_transform, target_transform=torch.tensor)
    test_dataset = datasets.SVHN(data_folder, split='test', download=True, transform=test_transform, target_transform=torch.tensor)    

nclasses = 10
nchannel = 3
mlp_dim = 1024 * 3
image_dim=32
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
# print(initial_train_indices)
np.random.shuffle(initial_train_indices)

# Get all indices except initially labelled set
non_init_idx = indices[~np.isin(indices, initial_train_indices)]
np.random.shuffle(non_init_idx)
# exit(0)

# Split the data into train and valid set (90 - 10 split)
split = int(np.floor(0.1 * num_train))
full_train_indices, valid_train_indices = non_init_idx[split:], non_init_idx[:split]
# pool_idx = np.random.choice(pool_idx, 30000)
np.random.shuffle(full_train_indices)
# remains = full_train_indices[30000:]
full_train_indices = full_train_indices[:30000]
print(len(np.unique(full_train_indices)))
# Dataloader for the initial labelled set, test set and validation set
# print(initial_train_indices, len(initial_train_indices))
# print(initial_train_indices)
init_subset = torch.utils.data.Subset(full_train_dataset, initial_train_indices)
pool_subset = torch.utils.data.Subset(full_train_dataset, full_train_indices)
valid_subset = torch.utils.data.Subset(full_train_dataset, valid_train_indices)
valid_queue = torch.utils.data.DataLoader(valid_subset, shuffle=False, batch_size=args.batch_size, num_workers=4)
valid_loader = DataLoader(valid_subset, shuffle=True, batch_size=64, num_workers=4)
train_queue = torch.utils.data.DataLoader(init_subset, shuffle=True, batch_size=args.batch_size, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
pool_loader = torch.utils.data.DataLoader(pool_subset, shuffle=False, batch_size=args.batch_size, num_workers=4)
   
switches = []
for i in range(14):
    switches.append([True for j in range(len(PRIMITIVES))])
switches_normal = copy.deepcopy(switches)
switches_reduce = copy.deepcopy(switches)
# To be moved to args
num_to_keep = [5, 3, 1]
num_to_drop = [3, 2, 2]
if len(args.add_width) == 3:
    add_width = args.add_width
else:
    add_width = [0, 0, 0]
if len(args.add_layers) == 3:
    add_layers = args.add_layers
else:
    add_layers = [0, 6, 12]
if len(args.dropout_rate) ==3:
    drop_rate = args.dropout_rate
else:
    drop_rate = [0.0, 0.0, 0.0]
eps_no_archs = [10, 10, 10] 
args = vars(args)
dt = data_train(Subset(full_train_dataset, initial_train_indices), None, args)
dt.train_nas(None, valid_loader=valid_loader, test_loader=test_loader, save_loc=save_loc, save_path=model_directory, nw="None")