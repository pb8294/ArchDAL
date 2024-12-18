# Data processing libraries
from distil.active_learning_strategies.jlp import JLPSampling
from distil.utils.train_eval import CustomImageDataset, label_propagation, label_propagation_orig, pretrain
import numpy as np

# Miscellaneous Libraries
import time
import math
import random
import os
import pickle

# Image libraries
import copy
from PIL import Image
import matplotlib.pyplot as plt
from checkpoint import Checkpoint, write_logs

# Torch Libraries
import torch
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import DataLoader

# Distil libraries for model and training
from distil.active_learning_strategies.random_sampling import RandomSampling
from distil.active_learning_strategies.margin_sampling import MarginSampling
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.core_set import CoreSet
from distil.active_learning_strategies.least_confidence_sampling import LeastConfidenceSampling
from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.bayesian_active_learning_disagreement_dropout import BALDDropout
from distil.active_learning_strategies.egl_sampling import EGLSampling
from distil.active_learning_strategies.entropy_sampling_dropout import EntropySamplingDropout
from distil.active_learning_strategies.margin_sampling_dropout import MarginSamplingDropout
from distil.active_learning_strategies.least_confidence_sampling_dropout import LeastConfidenceSamplingDropout

from distil.utils.train_helper import data_train
from distil.utils.utils import LabeledToUnlabeledDataset

from utils.util import plot_network_mask_all, plot_network_mask_base
from distil.utils.bbdrop.utils.misc import plot_network_mask_base as bbdrop_plot_network_mask_base

lLabels = {"MNIST": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], "FMNIST": ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], "SVHN": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

def train_one(full_train_dataset, initial_train_indices, valid_train_indices, test_loader, net, n_rounds, budget, args, nclasses, strategy, save_directory, checkpoint_directory, experiment_name, initial_seed_size=1000, nw=None):

    # Split the full training dataset into an initial training dataset and an unlabeled dataset
    train_dataset = Subset(full_train_dataset, initial_train_indices)
    valid_set = Subset(full_train_dataset, valid_train_indices)
    initial_unlabeled_indices = list(set(range(len(full_train_dataset))) - set(initial_train_indices)- set(valid_train_indices))
    unlabeled_dataset = Subset(full_train_dataset, initial_unlabeled_indices)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=64, num_workers=4)
    
    # Set up the AL strategy
    print(strategy)
    if strategy == "random":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "entropy":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "badge":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "coreset":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "bald":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == 'egl':
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = EGLSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, lLabels[args['dataset']], checkpoint_directory, strategy_args)
    elif strategy == "entropy_sampling":
        print("entropy_sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = EntropySamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence_sampling":
        print("lc sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = LeastConfidenceSamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin_sampling":
        print("margin sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = MarginSamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    # print()
    # elif strategy == "fass":
    #     strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
    #     strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    # elif strategy == "glister":
    #     strategy_args = {'batch_size' : args['batch_size'], 'lr': args['lr'], 'device':args['device']}
    #     strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args, typeOf='rand', lam=0.1)
    # elif strategy == "adversarial_bim":
    #     strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
    #     strategy = AdversarialBIM(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    # elif strategy == "adversarial_deepfool":
    #     strategy_args = {str'batch_size' : args['batch_size'], 'device':args['device']}
    #     strategy = AdversarialDeepFool(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)

    # Define acc initially
    acc = np.zeros(n_rounds+1)
    loss = np.zeros(n_rounds+1)
    pcount = np.zeros(n_rounds+1)

    initial_unlabeled_size = len(unlabeled_dataset)

    initial_round = 1

    # Define an index map
    index_map = np.array([x for x in range(initial_unlabeled_size)])

    # Attempt to load a checkpoint. If one exists, then the experiment crashed.
    # training_checkpoint = Checkpoint(experiment_name=experiment_name, path=checkpoint_directory)
    rec_acc, rec_loss, rec_indices, rec_state_dict = None, None, None, None #training_checkpoint.get_saved_values()

    # Check if there are values to recover
    if rec_acc is not None:

        # Restore the accuracy list
        for i in range(len(rec_acc)):
            acc[i] = rec_acc[i]
            loss[i] = rec_loss[i]

        # Restore the indices list and shift those unlabeled points to the labeled set.
        index_map = np.delete(index_map, rec_indices)

        # Record initial size of the training dataset
        intial_seed_size = len(train_dataset)

        restored_unlabeled_points = Subset(unlabeled_dataset, rec_indices)
        train_dataset = ConcatDataset([train_dataset, restored_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(rec_indices))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        # Restore the model
        net.load_state_dict(rec_state_dict) 

        # Fix the initial round
        initial_round = (len(train_dataset) - initial_seed_size) // budget + 1

        # Ensure loaded model is moved to GPU
        if torch.cuda.is_available():
            net = net.cuda()     

        strategy.update_model(net)
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset)) 

        dt = data_train(train_dataset, net, args)

    else:

        if torch.cuda.is_available():
            net = net.cuda()

        dt = data_train(train_dataset, net, args)

        # acc[0] = dt.get_acc_on_set(test_dataset)
        # Pradeep Bajracharya
        acc[0], loss[0] = dt.get_acc_on_set(test_loader, nw=nw)
        print('Initial Testing accuracy:', round(acc[0]*100, 2), 'Initial Testing loss:', round(loss[0]*100, 2), flush=True)

        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[0]*100, 2))
        logs['Test Loss'] =  str(round(loss[0], 2)) # Pradeep Bajracharya
        write_logs(logs, save_directory, 0)
          
        #Updating the trained model in strategy class
        strategy.update_model(net)

    # Record the training transform and test transform for disabling purposes
    train_transform = full_train_dataset.transform
    test_transform = test_loader.dataset.transform

    ##User Controlled Loop
    # clf.load_state_dict(torch.load(checkpoint_directory + "/m_latest.pt"))
    for rd in range(initial_round, n_rounds+1):
        print('-------------------------------------------------')
        print('Round', rd) 
        print('-------------------------------------------------')

        sel_time = time.time()
        full_train_dataset.transform = test_transform # Disable any augmentation while selecting points
        idx = strategy.select(budget)  
        # print(idx); exit()    
        full_train_dataset.transform = train_transform # Re-enable any augmentation done during training
        sel_time = time.time() - sel_time
        print("Selection Time:", sel_time)

        selected_unlabeled_points = Subset(unlabeled_dataset, idx)
        train_dataset = ConcatDataset([train_dataset, selected_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(idx))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        # Update the index map
        index_map = np.delete(index_map, idx, axis = 0)

        print('Number of training points -', len(train_dataset))

        # Start training
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
        dt.update_data(train_dataset)
        t1 = time.time()
        clf, train_logs = dt.train(None, valid_loader=valid_loader, save_loc=checkpoint_directory, save_path=checkpoint_directory + "/m_latest_rd.pt", nw=nw, rounds=True)

        clf.load_state_dict(torch.load(checkpoint_directory + "/m_latest_rd.pt"))
        dt.update_net(clf)
        if nw is not None:
            if args["network"] == "adacnn" or args["network"] == "resnetada":
                plot_network_mask_base(clf, save_directory + "/model_round"+str(rd)+".png", figsize=True)
                plot_network_mask_all(clf, checkpoint_directory + "/model_after_"+str(rd)+"-round.png")
            elif args["network"] == "adamlp":
                plot_network_mask_base(clf, save_directory + "/model_round"+str(rd)+".png", figsize=False)
        t2 = time.time()
        # acc[rd] = dt.get_acc_on_set(test_dataset)
        # Pradeep Bajracharya
        acc[rd], loss[rd] = dt.get_acc_on_set(test_loader, nw=nw)
        print(acc)
        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[rd]*100, 2))
        logs['Test Loss'] =  str(round(loss[rd], 2))
        logs['Selection Time'] = str(sel_time)
        logs['Trainining Time'] = str(t2 - t1) 
        logs['Training'] = train_logs
        write_logs(logs, save_directory, rd)
        strategy.update_model(clf)
        print('Testing accuracy:', round(acc[rd]*100, 2), ', Testing loss:', round(loss[rd]*100, 2), flush=True)
        # plot_network_mask_all(clf, checkpoint_directory + "/model_after_"+str(rd)+"-round.png")

        # Create a checkpoint
        used_indices = np.array([x for x in range(initial_unlabeled_size)])
        used_indices = np.delete(used_indices, index_map).tolist()
        # clf.get_param_count()
        # print("Param Count:" , clf.get_param_count())
        # round_checkpoint = Checkpoint(acc.tolist(), loss.tolist(), used_indices, clf.state_dict(), experiment_name=experiment_name)
        # round_checkpoint.save_checkpoint(checkpoint_directory, rd)
        # pcount[rd] = clf.get_param_count()
        # print(pcount)
        
    with open(os.path.join(checkpoint_directory,'param.txt'), 'w') as f:
        for item in pcount:
            f.write("%s\n" % item)
    print('Training Completed')
    return acc, loss

def train_one_bbdrop(full_train_dataset, initial_train_indices, valid_train_indices, test_loader, net, n_rounds, budget, args, nclasses, strategy, save_directory, checkpoint_directory, experiment_name, initial_seed_size=1000, nw=None):

    # Split the full training dataset into an initial training dataset and an unlabeled dataset
    train_dataset = Subset(full_train_dataset, initial_train_indices)
    valid_set = Subset(full_train_dataset, valid_train_indices)
    initial_unlabeled_indices = list(set(range(len(full_train_dataset))) - set(initial_train_indices)- set(valid_train_indices))
    unlabeled_dataset = Subset(full_train_dataset, initial_unlabeled_indices)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=64, num_workers=4)
    
    # Set up the AL strategy
    print(strategy)
    if strategy == "random":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "entropy":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "badge":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "coreset":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "bald":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == 'egl':
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = EGLSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, lLabels[args['dataset']], checkpoint_directory, strategy_args)
    elif strategy == "entropy_sampling":
        print("entropy_sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = EntropySamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence_sampling":
        print("lc sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = LeastConfidenceSamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin_sampling":
        print("margin sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = MarginSamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    # print()
    # elif strategy == "fass":
    #     strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
    #     strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    # elif strategy == "glister":
    #     strategy_args = {'batch_size' : args['batch_size'], 'lr': args['lr'], 'device':args['device']}
    #     strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args, typeOf='rand', lam=0.1)
    # elif strategy == "adversarial_bim":
    #     strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
    #     strategy = AdversarialBIM(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    # elif strategy == "adversarial_deepfool":
    #     strategy_args = {str'batch_size' : args['batch_size'], 'device':args['device']}
    #     strategy = AdversarialDeepFool(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)

    # Define acc initially
    acc = np.zeros(n_rounds+1)
    loss = np.zeros(n_rounds+1)
    pcount = np.zeros(n_rounds+1)

    initial_unlabeled_size = len(unlabeled_dataset)

    initial_round = 1

    # Define an index map
    index_map = np.array([x for x in range(initial_unlabeled_size)])

    # Attempt to load a checkpoint. If one exists, then the experiment crashed.
    # training_checkpoint = Checkpoint(experiment_name=experiment_name, path=checkpoint_directory)
    rec_acc, rec_loss, rec_indices, rec_state_dict = None, None, None, None #training_checkpoint.get_saved_values()

    # Check if there are values to recover
    if rec_acc is not None:

        # Restore the accuracy list
        for i in range(len(rec_acc)):
            acc[i] = rec_acc[i]
            loss[i] = rec_loss[i]

        # Restore the indices list and shift those unlabeled points to the labeled set.
        index_map = np.delete(index_map, rec_indices)

        # Record initial size of the training dataset
        intial_seed_size = len(train_dataset)

        restored_unlabeled_points = Subset(unlabeled_dataset, rec_indices)
        train_dataset = ConcatDataset([train_dataset, restored_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(rec_indices))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        # Restore the model
        net.load_state_dict(rec_state_dict) 

        # Fix the initial round
        initial_round = (len(train_dataset) - initial_seed_size) // budget + 1

        # Ensure loaded model is moved to GPU
        if torch.cuda.is_available():
            net = net.cuda()     

        strategy.update_model(net)
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset)) 

        dt = data_train(train_dataset, net, args)

    else:

        if torch.cuda.is_available():
            net = net.cuda()

        dt = data_train(train_dataset, net, args)

        # acc[0] = dt.get_acc_on_set(test_dataset)
        # Pradeep Bajracharya
        acc[0], loss[0] = dt.get_acc_on_set(test_loader, nw=nw)
        print('Initial Testing accuracy:', round(acc[0]*100, 2), 'Initial Testing loss:', round(loss[0]*100, 2), flush=True)

        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[0]*100, 2))
        logs['Test Loss'] =  str(round(loss[0], 2)) # Pradeep Bajracharya
        write_logs(logs, save_directory, 0)
          
        #Updating the trained model in strategy class
        strategy.update_model(net)

    # Record the training transform and test transform for disabling purposes
    train_transform = full_train_dataset.transform
    test_transform = test_loader.dataset.transform

    ##User Controlled Loop
    # clf.load_state_dict(torch.load(checkpoint_directory + "/m_latest.pt"))
    for rd in range(initial_round, n_rounds+1):
        print('-------------------------------------------------')
        print('Round', rd) 
        print('-------------------------------------------------')

        sel_time = time.time()
        full_train_dataset.transform = test_transform # Disable any augmentation while selecting points
        idx = strategy.select(budget)  
        # print(idx); exit()    
        full_train_dataset.transform = train_transform # Re-enable any augmentation done during training
        sel_time = time.time() - sel_time
        print("Selection Time:", sel_time)

        selected_unlabeled_points = Subset(unlabeled_dataset, idx)
        train_dataset = ConcatDataset([train_dataset, selected_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(idx))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        # Update the index map
        index_map = np.delete(index_map, idx, axis = 0)

        print('Number of training points -', len(train_dataset))

        # Start training
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
        dt.update_data(train_dataset)
        t1 = time.time()
        clf, train_logs = dt.train_bbdrop(None, valid_loader=valid_loader, save_loc=checkpoint_directory, save_path=checkpoint_directory + "/m_latest_rd.pt", nw=nw, rounds=True)

        clf.load_state_dict(torch.load(checkpoint_directory + "/m_latest_rd.pt"))
        dt.update_net(clf)
        if nw is not None:
            bbdrop_plot_network_mask_base(clf, save_directory + "/model_round"+str(rd)+".png", figsize=True)#; exit(0)

        t2 = time.time()
        # acc[rd] = dt.get_acc_on_set(test_dataset)
        # Pradeep Bajracharya
        acc[rd], loss[rd] = dt.get_acc_on_set(test_loader, nw=nw)
        print(acc)
        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[rd]*100, 2))
        logs['Test Loss'] =  str(round(loss[rd], 2))
        logs['Selection Time'] = str(sel_time)
        logs['Trainining Time'] = str(t2 - t1) 
        logs['Training'] = train_logs
        write_logs(logs, save_directory, rd)
        strategy.update_model(clf)
        print('Testing accuracy:', round(acc[rd]*100, 2), ', Testing loss:', round(loss[rd]*100, 2), flush=True)
        # plot_network_mask_all(clf, checkpoint_directory + "/model_after_"+str(rd)+"-round.png")

        # Create a checkpoint
        used_indices = np.array([x for x in range(initial_unlabeled_size)])
        used_indices = np.delete(used_indices, index_map).tolist()
        # clf.get_param_count()
        # print("Param Count:" , clf.get_param_count())
        # round_checkpoint = Checkpoint(acc.tolist(), loss.tolist(), used_indices, clf.state_dict(), experiment_name=experiment_name)
        # round_checkpoint.save_checkpoint(checkpoint_directory, rd)
        # pcount[rd] = clf.get_param_count()
        # print(pcount)
        
    with open(os.path.join(checkpoint_directory,'param.txt'), 'w') as f:
        for item in pcount:
            f.write("%s\n" % item)
    print('Training Completed')
    return acc, loss

def train_one_ssl(full_train_dataset, initial_train_indices, valid_train_indices, test_loader, net, n_rounds, budget, args, nclasses, strategy, save_directory, checkpoint_directory, experiment_name, initial_seed_size=1000, nw=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split the full training dataset into an initial training dataset and an unlabeled dataset
    train_dataset = Subset(full_train_dataset, initial_train_indices)
    valid_set = Subset(full_train_dataset, valid_train_indices)
    initial_unlabeled_indices = list(set(range(len(full_train_dataset))) - set(initial_train_indices)- set(valid_train_indices))
    unlabeled_dataset = Subset(full_train_dataset, initial_unlabeled_indices)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=64, num_workers=4)
    
    # Set up the AL strategy
    print(strategy)
    if strategy == "random":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "entropy":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "badge":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "coreset":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "bald":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == 'egl':
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        strategy = EGLSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, lLabels[args['dataset']], checkpoint_directory, strategy_args)
    elif strategy == "entropy_sampling":
        print("entropy_sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = EntropySamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence_sampling":
        print("lc sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = LeastConfidenceSamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin_sampling":
        print("margin sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = MarginSamplingDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "jlp":
        print("JLP sampling")
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device'], 'nw': nw}
        # strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, args=strategy_args, nw=nw)
        strategy = JLPSampling(train_dataset, unlabeled_dataset, net, nclasses, strategy_args)
        
    # Define acc initially
    acc = np.zeros(n_rounds+1)
    loss = np.zeros(n_rounds+1)
    pcount = np.zeros(n_rounds+1)

    initial_unlabeled_size = len(unlabeled_dataset)

    initial_round = 1

    # Define an index map
    index_map = np.array([x for x in range(initial_unlabeled_size)])

    # Attempt to load a checkpoint. If one exists, then the experiment crashed.
    # training_checkpoint = Checkpoint(experiment_name=experiment_name, path=checkpoint_directory)
    rec_acc, rec_loss, rec_indices, rec_state_dict = None, None, None, None #training_checkpoint.get_saved_values()

    # Check if there are values to recover
    if rec_acc is not None:

        # Restore the accuracy list
        for i in range(len(rec_acc)):
            acc[i] = rec_acc[i]
            loss[i] = rec_loss[i]

        # Restore the indices list and shift those unlabeled points to the labeled set.
        index_map = np.delete(index_map, rec_indices)

        # Record initial size of the training dataset
        intial_seed_size = len(train_dataset)

        restored_unlabeled_points = Subset(unlabeled_dataset, rec_indices)
        train_dataset = ConcatDataset([train_dataset, restored_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(rec_indices))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        # Restore the model
        net.load_state_dict(rec_state_dict) 

        # Fix the initial round
        initial_round = (len(train_dataset) - initial_seed_size) // budget + 1

        # Ensure loaded model is moved to GPU
        if torch.cuda.is_available():
            net = net.cuda()     

        strategy.update_model(net)
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset)) 

        dt = data_train(train_dataset, net, args)

    else:

        if torch.cuda.is_available():
            net = net.cuda()

        dt = data_train(train_dataset, net, args)

        # acc[0] = dt.get_acc_on_set(test_dataset)
        # Pradeep Bajracharya
        acc[0], loss[0] = dt.get_acc_on_set(test_loader, nw=nw)
        print('Initial Testing accuracy:', round(acc[0]*100, 2), 'Initial Testing loss:', round(loss[0]*100, 2), flush=True)

        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[0]*100, 2))
        logs['Test Loss'] =  str(round(loss[0], 2)) # Pradeep Bajracharya
        write_logs(logs, save_directory, 0)
          
        #Updating the trained model in strategy class
        strategy.update_model(net)

    # Record the training transform and test transform for disabling purposes
    train_transform = full_train_dataset.transform
    test_transform = test_loader.dataset.transform

    ##User Controlled Loop
    # clf.load_state_dict(torch.load(checkpoint_directory + "/m_latest.pt"))
    for rd in range(initial_round, n_rounds+1):
        print('-------------------------------------------------')
        print('Round', rd) 
        print('-------------------------------------------------')

        sel_time = time.time()
        full_train_dataset.transform = test_transform # Disable any augmentation while selecting points
        idx = strategy.select(budget)  
        # print(idx); exit()    
        full_train_dataset.transform = train_transform # Re-enable any augmentation done during training
        sel_time = time.time() - sel_time
        print("Selection Time:", sel_time)

        selected_unlabeled_points = Subset(unlabeled_dataset, idx)
        train_dataset = ConcatDataset([train_dataset, selected_unlabeled_points])
        
        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(idx))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        print('Number of training points -', len(train_dataset))

        # Start training
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
        
        # Label Propagation step
        lpp_st = time.time()
        print("Label Propagation Start ... ")
        new_labeled_idx = np.hstack([np.array(initial_train_indices), np.array(initial_unlabeled_indices)[np.array(idx)]])
        initial_unlabeled_indices = np.delete(initial_unlabeled_indices, idx)
        all_indices = np.hstack([new_labeled_idx, initial_unlabeled_indices])
        lu_init_indices = np.arange(len(new_labeled_idx))
        lu_pool_indices = np.arange(len(new_labeled_idx), len(all_indices))
        
        lu_train_dataset = Subset(full_train_dataset, all_indices)
        unshuffled_pre_trainloader = DataLoader(lu_train_dataset, shuffle=False, batch_size=64, num_workers=4)
        print("Obtaining features...")
        images_all, predictions_all, labels_all, X = pretrain(unshuffled_pre_trainloader, net, device, nw=nw)
        ft_end = time.time()
        print("Features obtained, time taken {}.".format(ft_end - lpp_st))
        labeled_idx = lu_init_indices
        unlabeled_idx = lu_pool_indices
        print("Label Propagation Step ... ")
        p_labels, weights, cweights = label_propagation_orig(X, labeled_idx, unlabeled_idx, labels_all)
        train_dataset_custom = CustomImageDataset(images_all, p_labels, weights=weights, cweights=cweights)
        print("Label Propagation end, time taken {}, total time: {}".format(time.time() - ft_end, time.time() - lpp_st))

        # Update the index map
        index_map = np.delete(index_map, idx, axis = 0)
        dt.update_data(train_dataset_custom)
        t1 = time.time()
        clf, train_logs = dt.train_ssl(None, valid_loader=valid_loader, save_loc=checkpoint_directory, save_path=checkpoint_directory + "/m_latest_rd.pt", nw=nw, rounds=True)

        clf.load_state_dict(torch.load(checkpoint_directory + "/m_latest_rd.pt"))
        dt.update_net(clf)
        if nw is not None:
            if args["network"] == "adacnn" or args["network"] == "resnetada":
                plot_network_mask_base(clf, save_directory + "/model_round"+str(rd)+".png", figsize=True)
                plot_network_mask_base(clf, checkpoint_directory + "/model_after_"+str(rd)+"-round.png")
            elif args["network"] == "adamlp":
                plot_network_mask_base(clf, save_directory + "/model_round"+str(rd)+".png", figsize=False)
        t2 = time.time()
        # acc[rd] = dt.get_acc_on_set(test_dataset)
        # Pradeep Bajracharya
        acc[rd], loss[rd] = dt.get_acc_on_set(test_loader, nw=nw)
        print(acc)
        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[rd]*100, 2))
        logs['Test Loss'] =  str(round(loss[rd], 2))
        logs['Selection Time'] = str(sel_time)
        logs['Trainining Time'] = str(t2 - t1) 
        logs['Training'] = train_logs
        write_logs(logs, save_directory, rd)
        strategy.update_model(clf)
        print('Testing accuracy:', round(acc[rd]*100, 2), ', Testing loss:', round(loss[rd]*100, 2), flush=True)
        # plot_network_mask_all(clf, checkpoint_directory + "/model_after_"+str(rd)+"-round.png")

        # Create a checkpoint
        used_indices = np.array([x for x in range(initial_unlabeled_size)])
        used_indices = np.delete(used_indices, index_map).tolist()
        
    with open(os.path.join(checkpoint_directory,'param.txt'), 'w') as f:
        for item in pcount:
            f.write("%s\n" % item)
    print('Training Completed')
    return acc, loss
