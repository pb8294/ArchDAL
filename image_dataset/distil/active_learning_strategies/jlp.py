from distil.utils.train_eval import pretrain
import torch

from .score_streaming_strategy import ScoreStreamingStrategy
from .strategy import Strategy

import faiss
import scipy
import numpy as np
import distil.utils.diffusion as diff

from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import DataLoader

class JLPSampling(Strategy):
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(JLPSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
    def select(self, budget):
        k = 50
        labeled_idxs = range(len(self.labeled_dataset))
        all_dataset = ConcatDataset([self.labeled_dataset, self.unlabeled_dataset])
        unshuffled_pre_trainloader = DataLoader(all_dataset, shuffle=False, batch_size=64, num_workers=4)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Extract features
        _, _, labels_all, X = pretrain(unshuffled_pre_trainloader, self.model, device)
        # X, _, preds = extract_features(dataset, self.model)

        # Perform Diffusion
        alpha = 0.99
        d = X.shape[1]
        labels = np.asarray(labels_all)
        labeled_idx = np.asarray(labeled_idxs)

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res,d,flat_config)

        faiss.normalize_L2(X)
        index.add(X)
        N = X.shape[0]
        D, I = index.search(X, k + 1)
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T

        Wknn = scipy.sparse.csr_matrix((D.flatten('F'),
                                       (row_idx_rep.flatten('F'), I.flatten('F'))),
                                       shape=(N, N))

        qsim = np.zeros((N))
        for i in range(10):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            qsim[cur_idx] = 1.0 #/ cur_idx.shape[0]

        W = Wknn
        W = W + W.T

        Wn = diff.normalize_connection_graph(W)
        Wnn = np.eye(Wn.shape[0]) - alpha * Wn
        selected_indices = list()

        # Perform diffusion after adding the newly selected images
        for x in range(budget):
            print(x)
            cg_ranks, cg_sims =  diff.cg_diffusion_sel(qsim, Wnn, tol = 1e-6)

            cg_sims[labeled_idx] = 1.0
            if selected_indices:
                cg_sims[selected_indices] = 1.0
            ranks = np.argsort(cg_sims, axis = 0)

            it = 0
            while True:
                sel_id = ranks[it]
                it += 1
                if sel_id not in selected_indices and sel_id not in labeled_idx:
                    break

            qsim[sel_id] = 1.0
            selected_indices.append(sel_id)

        print(selected_indices); exit(0)
        return selected_indices