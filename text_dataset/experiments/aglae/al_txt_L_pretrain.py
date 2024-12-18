import argparse
import os
import pickle
import time
import json
import logging

import torch
import torch.nn as nn
import hydra
import math
import transformers

import lightning as L
import wandb

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/pb8294/Projects/dal-toolbox-main")
from dal_toolbox import datasets
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox import metrics
from dal_toolbox.models.utils.callbacks import MetricLogger
from utils import build_dataset, build_model, build_query, initialize_wandb, strategy_results
from lightning.pytorch.callbacks import LearningRateMonitor
import numpy as np
from torch.utils.data import Subset, DataLoader
from datasets import Dataset
import faiss

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

@hydra.main(version_base=None, config_path="./configs", config_name="al_nlp")
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # print(args); exit(0)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    args.output_dir = os.path.expanduser(args.output_dir)
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # print('%spretrained_%s_%s_lr_%s_batch_%s_%s_final.pickle' % (args.output_dir, args.dataset.name, args.model.name, 1, args.model.batch_size, 300)); exit(0)

    # logging
    results = {}
    queried_indices = {}
    # initialize_wandb(args)

    # Setup Data
    logging.info('Building dataset %s', args.dataset.name)
    data = build_dataset(args)
    collator = transformers.DataCollatorWithPadding(
        tokenizer=data.tokenizer,
        padding='longest',
        return_tensors='pt'
    )
    
    
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        query_dataset=data.query_dataset,
        val_dataset=data.val_dataset,
        train_batch_size=args.model.batch_size,
        predict_batch_size=args.model.batch_size*4,
        collator=collator
    )

    if args.al_cycle.init_pool_file is not None: 
        logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
        with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
            initial_indices = json.load(f)
        assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match'
        al_datamodule.update_annotations(initial_indices)
    else:
        logging.info('Creating random initial labeled pool with %s samples.', args.al_cycle.n_init)
        al_datamodule.random_init(n_samples=args.al_cycle.n_init)

    queried_indices['cycle0'] = sorted(al_datamodule.labeled_indices)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=data.num_classes, len_trainset=len(data.train_dataset))
    # print(model)
    # print(list(model.state_dict().keys()))
    # print(args.output_dir)
    # exit(0)
    
    test_dataloader = DataLoader(
        data.test_dataset,
        batch_size=args.model.batch_size*4,
        shuffle=False,
        collate_fn=collator
    )
    

    # print(data.test_dataset["label"])
    # data.set_label("test", np.ones(7600))
    # print(data.test_dataset["label"])
    # exit(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(data.test_dataset)
    all_loader = DataLoader(
                data.train_dataset,
                batch_size=args.model.batch_size*4,
                shuffle=False,
                collate_fn=collator
            )

    save_every = 1
    # if args.pretrain == 100:
    if args.al_cycle.cold_start:
        model.reset_states()
        
    
        
    for epoch_n in range(1, 6):
        # print(epoch_n % save_every == 0, epoch_n > 0)
        # exit(0)
        reprs, inps, masks = model.get_representation_input(all_loader)
        
        processed_e = preprocess_features(reprs.numpy())
        print(processed_e.shape)
        st = time.time()
        km_I, _ = run_kmeans(processed_e, data.num_classes)
        et = time.time()
        print("K_means", et - st)
        lists = []
        for inp in inps:
            for x in inp:
                lists.append(x)
                
        lists_msk = []
        for inp in masks:
            for x in inp:
                lists_msk.append(x)
        
        km_I = torch.tensor(km_I)
        print(km_I)
        diction = {
            "label": km_I,#data.test_dataset["label"],#np.ones(len(lists)),
            "input_ids": lists,
            "attention_mask": lists_msk
        }
    
        
        dataset = Dataset.from_dict(diction)
     
        all_loader_pretrain = DataLoader(
            dataset,
            batch_size=args.model.batch_size*4,
            shuffle=False,
            collate_fn=collator
        )
        
        
        al_datamodule = ActiveLearningDataModule(
            train_dataset=dataset,
            train_batch_size=args.model.batch_size,
            predict_batch_size=args.model.batch_size*4,
            collator=collator
        )
        model.train()
        model.lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=model.optimizer,
            num_warmup_steps=math.ceil(args.model.n_epochs * len(al_datamodule.train_dataloader()) * args.model.optimizer.warmup_ratio),
            num_training_steps=args.model.n_epochs * len(al_datamodule.train_dataloader())
        )
        
        trainer = L.Trainer(
            devices=1,
            accelerator="gpu",
            max_epochs=1)

        print(model)
        
        trainer.fit(model, all_loader_pretrain)
    
        
        if epoch_n % save_every == 0 and (epoch_n > 0):
            here = os.path.dirname(os.path.abspath(__file__))
            print("saving...", os.path.join(here, './pretrained_{}_{}_final.pickle'.format(args.dataset.name, args.model.name)))
            stored_model_keys = list(model.state_dict().keys())[:-2]
            msd =  model.state_dict()
            pretrained_dict = {key:msd[key] for key in stored_model_keys}
            
            with open(os.path.join(here, './pretrained_{}_{}_final.pickle'.format(args.dataset.name, args.model.name)), 'wb') as handle:
                print('pickle file opened')
                pickle.dump(pretrained_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # exit(0)
        
def evaluate_cycle(logits, targets, args):
    test_stats = {}

    if args.dataset.n_classes <= 2:
        test_f1_macro = metrics.f1_macro(logits, targets, 'binary')
        test_f1_micro = test_f1_macro
    else:
        test_f1_macro = metrics.f1_macro(logits, targets, 'macro')
        test_f1_micro = metrics.f1_macro(logits, targets, 'micro')
    
    test_stats.update({
        "test_acc": metrics.Accuracy()(logits, targets).item(),
        "test_f1_macro": test_f1_macro,
        "test_f1_micro": test_f1_micro,
        "test_acc_blc": metrics.balanced_acc(logits, targets)
    })
        
    return test_stats

if __name__ == '__main__':
    main()







def auc_results(logits, targets):
    pass
    



#%%

    # if args.al_cycle.init_pool_file is not None:
    #     logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
    #     with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
    #         initial_indices = json.load(f)
    #     assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match.'
    #     al_dataset.update_annotations(initial_indices)
