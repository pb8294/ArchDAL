from .strategy import Strategy
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import stats


def kmeanspp(X, mags=None, K=1000):
    lengths = np.linalg.norm(X, 2, axis=1)
    if mags is None:
        ind = np.argmax(lengths)
    else:
        ind = np.argmax(mags)
    indmin = np.argmin(lengths)
    indmax = np.argmin(lengths)
    X = X / lengths[:, None]
    # print(np.linalg.norm(X, axis=1))
    # exit(0)
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        # if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        # print(D2.max(), D2.argmax(), Ddist.max(), Ddist.argmax(), D2[indmin], Ddist[indmin])
        # exit(0)
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return np.array(indsAll)
    
    
class EGLSampling(Strategy):
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, llabel, save_path, args={}):
        
        super(EGLSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.llabel = self.llabel
        self.save_path = self.save_path
    
    def select(self, budget):
        all_grad, all_g_norm = self.egl_hyperparameter(self.unlabeled_dataset, self.llabel, data_save_path=self.save_path)
        idx_kpp = self.kmeanspp(all_grad, mags=all_g_norm, K=budget)
        return idx_kpp
        
    
    