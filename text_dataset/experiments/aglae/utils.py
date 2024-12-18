import wandb
import torch
import torch.nn as nn
import transformers
import math
from omegaconf import OmegaConf

import os
os.environ['WANDB_DIR'] = os.getcwd() + "/wandb/"
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/wandb/.cache/"
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + "/wandb/.config/"

from dal_toolbox.metrics.generalization import area_under_curve
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge
from dal_toolbox.datasets import activeglae
from dal_toolbox.models import deterministic
from dal_toolbox import metrics

def strategy_results(results):    
    acc = [cycles['test_stats']['test_acc'] for cycles in results.values()]
    f1_macro = [cycles['test_stats']['test_f1_macro'] for cycles in results.values()]
    f1_micro = [cycles['test_stats']['test_f1_micro'] for cycles in results.values()]
    acc_blc = [cycles['test_stats']['test_acc_blc'] for cycles in results.values()]

    auc_acc = area_under_curve(acc)
    auc_f1_macro = area_under_curve(f1_macro)
    auc_f1_micro = area_under_curve(f1_micro)
    auc_acc_blc = area_under_curve(acc_blc)

    auc_results = {
        'final_auc_acc': auc_acc,
        'final_auc_f1_macro': auc_f1_macro,
        'final_auc_f1_micro': auc_f1_micro,
        'final_auc_acc_blc': auc_acc_blc 
    }
    wandb.log(auc_results)
    results.update(auc_results)
    

    final_results = {
        'final_acc': acc[-1],
        'final_f1_macro': f1_macro[-1],
        'final_f1_micro': f1_micro[-1],
        'final_acc_blc': f1_micro[-1]
    }
    wandb.log(final_results)
    results.update(final_results)

    return results


def initialize_wandb(args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="aglae-project",
        dir="/home/pb8294/Projects/dal-toolbox-main/",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        }
    )

def build_dataset(args):
    if args.dataset.name == 'agnews':
        data = activeglae.AGNews(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )
    
    elif args.dataset.name == 'banks77':
        data = activeglae.Banks77(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'dbpedia':
        data = activeglae.DBPedia(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'fnc1':
        data = activeglae.FNC1(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'mnli':
        data = activeglae.MNLI(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'qnli':
        data = activeglae.QNLI(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'sst2':
        data = activeglae.SST2(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'trec6':
        data = activeglae.TREC6(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'wikitalk':
        data = activeglae.Wikitalk(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    elif args.dataset.name == 'yelp5':
        data = activeglae.Yelp5(
            model_name=args.model.name_hf,
            dataset_path=args.dataset_path,
            val_split=args.val_split
        )

    else:
        raise NotImplementedError(f'Dataset {args.dataset.name} not implemented.')
    
    return data 

def build_model(args, **kwargs):
    num_classes = args.dataset.n_classes
    len_trainset = kwargs['len_trainset']

    if args.model.name == 'bert':
        base_model = deterministic.bert.BertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
        )

        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay
        )

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.model.n_epochs * len_trainset * args.model.optimizer.warmup_ratio),
            num_training_steps=args.model.n_epochs * len_trainset
        )
        
        model = deterministic.DeterministicAGLAEModel(
            model=base_model,
            loss_fn=nn.CrossEntropyLoss(), 
            optimizer=optimizer,
            lr_scheduler=scheduler,
            scheduler_interval='step',
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics=None
        )
        
    elif args.model.name == 'roberta':
        base_model = deterministic.roberta.RoBertaSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
        )

        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay
        )

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.model.n_epochs * len_trainset * args.model.optimizer.warmup_ratio),
            num_training_steps=args.model.n_epochs * len_trainset
        )
        
        model = deterministic.DeterministicAGLAEModel(
            model=base_model,
            loss_fn=nn.CrossEntropyLoss(), 
            optimizer=optimizer,
            lr_scheduler=scheduler,
            scheduler_interval='step',
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics=None
        )

    
    elif args.model.name == 'distilbert':
        base_model = deterministic.distilbert.DistilbertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
        )

        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay
        )

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.model.n_epochs * len_trainset * args.model.optimizer.warmup_ratio),
            num_training_steps=args.model.n_epochs * len_trainset
        )
        
        model = deterministic.DeterministicAGLAEModel(
            model=base_model,
            loss_fn=nn.CrossEntropyLoss(), 
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics=None
        )
   
    else:
        raise NotImplementedError (f'Model {args.model.name} not implemented.')
    
    return model 

def build_query(args, **kwargs):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()

    elif args.al_strategy.name == "least_confident":
        query = uncertainty.LeastConfidentSampling(subset_size=args.dataset.train_subset)

    elif args.al_strategy.name == "margin":
        query = uncertainty.MarginSampling(subset_size=args.dataset.train_subset)

    elif args.al_strategy.name == "entropy":
        query = uncertainty.EntropySampling(subset_size=args.dataset.train_subset)

    elif args.al_strategy.name == "bayesian_entropy":
        query = uncertainty.BayesianEntropySampling(subset_size=args.dataset.train_subset)

    elif args.al_strategy.name == 'variation_ratio':
        query = uncertainty.VariationRatioSampling(subset_size=args.dataset.train_subset)

    elif args.al_strategy.name == 'bald':
        query = uncertainty.BALDSampling(subset_size=args.dataset.train_subset)

    elif args.al_strategy.name == "coreset":
        query = coreset.CoreSet(subset_size=args.dataset.train_subset)
        
    elif args.al_strategy.name == "badge":
        query = badge.Badge(subset_size=args.dataset.train_subset)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query