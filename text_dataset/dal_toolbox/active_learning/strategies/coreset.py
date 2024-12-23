import torch

from torch.utils.data import DataLoader

from .query import Query


class CoreSet(Query):
    def __init__(self, subset_size=None):
        super().__init__()
        self.subset_size = subset_size

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_dataloader, _ = al_datamodule.labeled_dataloader()

        features_unlabeled = model.get_representation(unlabeled_dataloader)
        features_labeled = model.get_representation(labeled_dataloader)

        chosen = self.kcenter_greedy(features_unlabeled, features_labeled, acq_size)
        return [unlabeled_indices[idx] for idx in chosen]

    def kcenter_greedy(self, features_unlabeled: torch.Tensor, features_labeled: torch.Tensor, acq_size: int):
        n_unlabeled = len(features_unlabeled)

        distances = torch.cdist(features_unlabeled, features_labeled)
        min_dist, _ = torch.min(distances, axis=1)

        idxs = []
        for _ in range(acq_size):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = torch.cdist(features_unlabeled, features_unlabeled[idx].unsqueeze(0))
            for j in range(n_unlabeled):
                min_dist[j] = torch.min(min_dist[j], dist_new_ctr[j, 0])
        return idxs
