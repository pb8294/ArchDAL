import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import faiss
import scipy
import distil.utils.diffusion as diff
import numpy as np
import torch.nn.functional as F

def pretrain(trainloader, net, device, nw=None):
    f_st = time.time()
    net.eval()
    images_all, predictions_all, labels_all, embeddings_all = [], [], [], []

    for batch_idx, batch in enumerate(trainloader):

        # Data.
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)


        # Prediction.
        preds, feats = net(inputs, last=True)
        images_all.append(inputs.data.cpu())
        labels_all.append(labels.data.cpu())
        if nw is None:
            predictions_all.append(preds.data.cpu())
            embeddings_all.append(feats.data.cpu())
        else:
            predictions_all.append(preds.data.cpu().mean(0))
            embeddings_all.append(feats.data.cpu().mean(0))
        # print(preds.mean(0).shape, feats.mean(0).shape)

    images_all = torch.cat(images_all)
    predictions_all = torch.cat(predictions_all)
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    labels_all = torch.cat(labels_all).numpy()
    f_et = time.time()
    return images_all, predictions_all, labels_all, embeddings_all

def train(trainloader, model, optimizer, epoch, device, nw=None):
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # print(epoch, i)
        inputs, labels, _, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        if nw is None:
            loss = criterion(outputs, labels)
        elif nw:
            # outputs = outputs.mean(0)
            # loss = criterion(outputs, labels)
            criterion = criterion.to(device)
            # print(len(trainloader.dataset)); exit()
            loss, kls, kl_scale, e_total= model.estimate_ELBO(criterion, outputs, labels, len(trainloader.dataset), kl_weight=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)

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

def label_propagation(X, labeled_idx, unlabeled_idx, labels_all, w_criterion="entropyl1", w_mode=True, temp=1, k=50):
    alpha = 0.99
    w_criterion = "entropyl1"
    y_pooling="mean"
    
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res,d,flat_config)

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)
    # print(D.shape, I.shape)
    # Create the graph
    D = D[:,1:] ** 3
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    Wknn = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'),
                                    I.flatten('F'))), shape=(N, N))

    W = Wknn
    W = W + W.T

    Wn = diff.normalize_connection_graph(W)

    # Initiliaze the y vector for each class and apply label propagation
    qsim = np.zeros((10,N))
    for i in range(10):
        cur_idx = labeled_idx[np.where(labels_all[labeled_idx] == i)]
        if y_pooling == 'mean':
            qsim[i,cur_idx] = 1.0 / cur_idx.shape[0]
        elif y_pooling == 'sum':
            qsim[i,cur_idx] = 1.0
        else:
            raise ValueError("y_pooling method not defined.")

    cg_ranks, cg_sims =  diff.cg_diffusion(qsim, Wn, alpha , tol = 1e-6)

    if temp > 0:
        cg_sims_temp = cg_sims * temp

    p_labels = np.argmax(cg_sims,1)
    probs = F.softmax(torch.tensor(cg_sims)).numpy()
    probs_temp = F.softmax(torch.tensor(cg_sims_temp)).numpy()
    probs_temp[probs_temp <0] = 0

    prob_sort = np.amax(probs_temp, axis=1)
    p_labels = np.argmax(probs_temp,1)

    if w_criterion == "entropy":
        entropy = scipy.stats.entropy(probs_temp.T)
        weights = 1 - entropy / np.log(10)
    elif w_criterion == 'entropyl1':
        cg_sims[cg_sims < 0] = 0
        probs_l1 = F.normalize(torch.tensor(cg_sims),1).numpy()
        probs_l1[probs_l1 <0] = 0

        probs_temp = F.softmax(torch.tensor(probs_l1 * temp)).numpy()

        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(10)
        weights = weights / np.max(weights)

        p_labels = np.argmax(probs_l1,1)

    else:
        raise ValueError("weight criterion not defined.")

    correct_idx = (p_labels == labels_all)
    acc = correct_idx.mean()

    p_labels[labeled_idx] = labels_all[labeled_idx]
    weights[labeled_idx] = 1.0
    
    return p_labels, weights

def label_propagation_orig(X, labeled_idx, unlabeled_idx, labels_all, k=50,  max_iter = 20):
    alpha = 0.99
    
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res,d,flat_config)

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)
    # print(D.shape, I.shape)
    # Create the graph
    D = D[:,1:] ** 3
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N,10))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(10):
        cur_idx = labeled_idx[np.where(labels_all[labeled_idx] ==i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:,i] = f

    # Handle numberical errors
    Z[Z < 0] = 0 

    probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
    probs_l1[probs_l1 <0] = 0
    entropy = scipy.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(10)
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1,1)

    correct_idx = (p_labels == labels_all)
    acc = correct_idx.mean()

    p_labels[labeled_idx] = labels_all[labeled_idx]
    weights[labeled_idx] = 1.0
    
    class_weights = np.ones((10,),dtype = np.float32)
    for i in range(10):
        cur_idx = np.where(np.asarray(p_labels) == i)[0]
        class_weights[i] = (float(labels_all.shape[0]) / 10) / cur_idx.size
    
    return p_labels, weights, class_weights


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, weights=None, cweights=None):
        self.images = images
        self.labels = labels
        if weights is None:
            self.weights = [1] * len(self.labels)
        else:
            self.weights = weights
            
        if cweights is None:
            self.cweights = np.ones((10,),dtype = np.float32)
        else:
            self.cweights = cweights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label, weight = self.images[idx], self.labels[idx], self.weights[idx]
        cweight = self.cweights[label]
        return image, label, weight, cweight