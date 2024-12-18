import copy
import logging
import os
import pickle
import time
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import torch.optim as optim
import sys
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from utils.models.genotypes import PRIMITIVES, Genotype
from utils.models.model_search import Network
# from utils.visualize import plot

from utils.util import plot_network_mask_all, plot_network_mask_base, frange_cycle_cosine, frange_cycle_linear, frange_cycle_sigmoid, geometric_discount
import distil.utils.utils_sub as utils

sys.path.append('../')  

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class AddIndexDataset(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        if len(self.wrapped_dataset[index]) == 2:
            data, label = self.wrapped_dataset[index]
            return data, label, index
        elif len(self.wrapped_dataset[index]) > 2:
            data, label, wt, cwt = self.wrapped_dataset[index]
            return data, label, wt, cwt, index
    
    def __len__(self):
        return len(self.wrapped_dataset)

#custom training
class data_train:

    """
    Provides a configurable training loop for AL.
    
    Parameters
    ----------
    training_dataset: torch.utils.data.Dataset
        The training dataset to use
    net: torch.nn.Module
        The model to train
    args: dict
        Additional arguments to control the training loop
        
        `batch_size` - The size of each training batch (int, optional)
        `islogs`- Whether to return training metadata (bool, optional)
        `optimizer`- The choice of optimizer. Must be one of 'sgd' or 'adam' (string, optional)
        `isverbose`- Whether to print more messages about the training (bool, optional)
        `isreset`- Whether to reset the model before training (bool, optional)
        `max_accuracy`- The training accuracy cutoff by which to stop training (float, optional)
        `min_diff_acc`- The minimum difference in accuracy to measure in the window of monitored accuracies. If all differences are less than the minimum, stop training (float, optional)
        `window_size`- The size of the window for monitoring accuracies. If all differences are less than 'min_diff_acc', then stop training (int, optional)
        `criterion`- The criterion to use for training (typing.Callable[], optional)
        `device`- The device to use for training (string, optional)
    """
    
    def __init__(self, training_dataset, net, args):
        self.train_set = training_dataset
        self.training_dataset = AddIndexDataset(training_dataset)
        self.net = net
        self.args = args
        
        self.n_pool = len(training_dataset)
        
        if 'islogs' not in args:
            self.args['islogs'] = False

        if 'optimizer' not in args:
            self.args['optimizer'] = 'sgd'
        
        if 'isverbose' not in args:
            self.args['isverbose'] = False
        
        if 'isreset' not in args:
            self.args['isreset'] = True

        if 'max_accuracy' not in args:
            self.args['max_accuracy'] = 0.95

        if 'min_diff_acc' not in args: #Threshold to monitor for
            self.args['min_diff_acc'] = 0.001

        if 'window_size' not in args:  #Window for monitoring accuracies
            self.args['window_size'] = 10
            
        if 'criterion' not in args:
            self.args['criterion'] = nn.CrossEntropyLoss()
            
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def update_data(self, new_training_dataset):
        """
        Updates the training dataset with the provided new training dataset
        
        Parameters
        ----------
        new_training_dataset: torch.utils.data.Dataset
            The new training dataset
        """
        self.training_dataset = AddIndexDataset(new_training_dataset)

    # Pradeep Bajracharya
    def update_net(self, new_net):
        """
        Updates the model with the provided new model
        
        Parameters
        ----------
        new_net: 
            The new network
        """
        self.net = new_net

    def get_acc_on_set(self, loader_te, shuffle=1, nw=None, epoch=None):
        
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
        try:
            self.clf
        except:
            self.clf = self.net


        if loader_te is None:
            raise ValueError("Test loader not present")
        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1 
        
        # if shuffle:
        #     loader_te = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=batch_size)
        # else:
        #     loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)
        self.clf.eval()
        accFinal = 0.
        
        # Pradeep Bajracharya
        lossFinal = 0.
        loglike = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"
        
        y_actual = None
        y_predic = None		

        with torch.no_grad():        
            # self.clf = self.clf.to(device=self.device)
            for batch_id, (x,y) in enumerate(loader_te):     
                x, y = x.float().to(device=self.device), y.to(device=self.device)
                # print(y[-20:-15]); exit(0)

                if nw == "ada1":
                    out = self.clf(x, 2)
                else:
                    out = self.clf(x)
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
        # exit()
        # print(acc_final)
        # acc_final = accFinal / len(test_dataset)
        # return accFinal / len(test_dataset)
        # Pradeep Bajracharya
        return acc_final, lossFinal / len(loader_te)

    def _train_weighted(self, epoch, loader_tr, optimizer, gradient_weights):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "none"

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)
            gradient_weights = gradient_weights.to(device=self.device)

            optimizer.zero_grad()
            out = self.clf(x)

            # Modify the loss function to apply weights before reducing to a mean
            loss = criterion(out, y.long())

            # Perform a dot product with the loss vector and the weight vector, then divide by batch size.
            weighted_loss = torch.dot(loss, gradient_weights[idxs])
            weighted_loss = torch.div(weighted_loss, len(idxs))

            accFinal += torch.sum(torch.eq(torch.max(out,1)[1],y)).item() #.data.item()

            # Backward now does so on the weighted loss, not the regular mean loss
            weighted_loss.backward() 

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset), weighted_loss

    
    def _train(self, epoch, loader_tr, optimizer, nw=None, weight=1, opt_for=None, valid_set=None):
        self.clf.train()
        accFinal = 0.
        # Pradeep Bajracharya
        lossFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"
        klFinal = 0.
        klScale = 0.
        eLog = 0.
        accs_arr = []
        train_accs = 0.
        # print("Epoch: ", epoch, "Nw:", nw, "Opt for:", opt_for)
        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            
            x, y = x.float().to(device=self.device), y.to(device=self.device)
            
            optimizer.zero_grad()
            # print(x.shape)
            out = self.clf(x)
            # print(out[-1], y[-1])
            if nw is None:
                # print("hh")
                loss = criterion(out, y.long())
                kls, kl_scale, e_total = 0.0, 0.0, 0.0
            else:
                # print("Not here")
                # loss = criterion(out, y.long())
                # kls, kl_scale, e_total = 0.0, 0.0, 0.0
                loss, kls, kl_scale, e_total= self.clf.estimate_ELBO(criterion, out, y, len(loader_tr.dataset), kl_weight=weight)
                # loss = criterion(out, y.long())
            # print(out.shape, opt_for)
            if nw == 'ada1':
                out = out.mean(0)
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            with torch.no_grad():
                y_pred = torch.max(out,1)[1]
                
                accFinal += torch.sum(y_pred == y).float().item()
            lossFinal += loss.item()
            if nw is not None:
                klFinal += kls.item()
                klScale += kl_scale
                eLog += e_total.item()
        # print(accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr)); exit()
        # print(loss.item(), lossFinal, lossFinal / len(loader_tr), len(loader_tr.dataset), batch_id)
        # return accFinal / len(loader_tr.dataset), loss
        return accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr), klFinal / len(loader_tr), klScale / len(loader_tr), eLog / len(loader_tr)
    
    def _train_ssl(self, epoch, loader_tr, optimizer, nw=None, weight=1, opt_for=None, valid_set=None):
        self.clf.train()
        accFinal = 0.
        # Pradeep Bajracharya
        lossFinal = 0.
        criterion =  nn.CrossEntropyLoss(size_average=False, ignore_index=-1, reduce=False).cuda()
        # criterion.reduction = "mean"
        klFinal = 0.
        klScale = 0.
        eLog = 0.
        accs_arr = []
        train_accs = 0.
        # print("Epoch: ", epoch, "Nw:", nw, "Opt for:", opt_for)
        for batch_id, (x, y, wt, cwt, idxs) in enumerate(loader_tr):
            
            x, y = x.float().to(device=self.device), y.to(device=self.device)
            wt = wt.to(device=self.device)
            cwt = cwt.to(device=self.device)
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)
            wt = torch.autograd.Variable(wt)
            cwt = torch.autograd.Variable(cwt)
            optimizer.zero_grad()
            # print(x.shape)
            out = self.clf(x)
            # print(out[-1], y[-1])
            if nw is None:
                # print("hh")
                loss = criterion(out, y.long())
                loss = loss * wt.float()
                loss = loss * cwt
                loss = loss.sum() / len(loader_tr)
                if torch.isnan(loss):
                    print(loss.item(), out, y.long(), cwt, wt); exit(0)
                kls, kl_scale, e_total = 0.0, 0.0, 0.0
            else:
                # print("Not here")
                # loss = criterion(out, y.long())
                # kls, kl_scale, e_total = 0.0, 0.0, 0.0
                # out = out.mean(0)
                e_total = criterion(out.mean(0), y.long())
                e_total = e_total * wt.float()
                e_total = e_total * cwt
                e_total = e_total.sum() / len(loader_tr)
                kls = self.clf.structure_sampler.get_kl()
                loss = e_total + kls / len(loader_tr.dataset)
                # loss, kls, kl_scale, e_total= self.clf.estimate_ELBO(criterion, out, y, len(loader_tr.dataset), kl_weight=weight)
                # loss = criterion(out, y.long())
            # print(out.shape, opt_for)
            if nw == 'ada1':
                out = out.mean(0)
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            with torch.no_grad():
                y_pred = torch.max(out,1)[1]
                
                accFinal += torch.sum(y_pred == y).float().item()
            lossFinal += loss.item()
            if nw is not None:
                klFinal += kls.item()
                klScale += 1
                eLog += e_total.item()
        # print(accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr)); exit()
        # print(loss.item(), lossFinal, lossFinal / len(loader_tr), len(loader_tr.dataset), batch_id)
        # return accFinal / len(loader_tr.dataset), loss
        return accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr), klFinal / len(loader_tr), klScale / len(loader_tr), eLog / len(loader_tr)
    
    def _train_bbdrop(self, epoch, loader_tr, optimizer, nw=None, weight=1, opt_for=None, valid_set=None):
        self.clf.train()
        accFinal = 0.
        # Pradeep Bajracharya
        lossFinal = 0.
        criterion = self.args['criterion']
        # criterion.reduction = "mean"
        klFinal = 0.
        klScale = 0.
        eLog = 0.
        accs_arr = []
        train_accs = 0.
        # print("Epoch: ", epoch, "Nw:", nw, "Opt for:", opt_for)
        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            
            x, y = x.float().to(device=self.device), y.to(device=self.device)
            
            optimizer.zero_grad()
            # print(x.shape)
            out = self.clf(x)
            # print(out[-1], y[-1])
            cent = criterion(out, y)
            kls = self.clf.get_reg().cuda()
            loss = cent + 1./60000 * kls
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            with torch.no_grad():
                y_pred = torch.max(out,1)[1]
                
                accFinal += torch.sum(y_pred == y).float().item()
            lossFinal += loss.item()
            klFinal += kls.item()
            klScale += 1./60000
            eLog += cent.item()
        # print(accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr)); exit()
        # print(loss.item(), lossFinal, lossFinal / len(loader_tr), len(loader_tr.dataset), batch_id)
        # return accFinal / len(loader_tr.dataset), loss
        return accFinal / len(loader_tr.dataset), lossFinal / len(loader_tr), klFinal / len(loader_tr), klScale / len(loader_tr), eLog / len(loader_tr)

    def check_saturation(self, acc_monitor):
        
        saturate = True

        for i in range(len(acc_monitor)):
            for j in range(i+1, len(acc_monitor)):
                if acc_monitor[j] - acc_monitor[i] >= self.args['min_diff_acc']:
                    saturate = False
                    break

        return saturate

    def train(self, gradient_weights=None, valid_loader=None, test_loader=None, save_loc="./", save_path="./model.pt", nw=None, rounds=False):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        # v_acc, v_loss = self.get_acc_on_set(valid_set, nw=None)
        # print(v_acc, v_loss); exit()
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        n_epoch = self.args['n_epoch']
        # self.args['isreset'] = False
        self.clf = self.net.to(device=self.device)
        if self.args['isreset']:
            if nw is not None:
                self.clf.structure_sampler.reset_parameters()
            self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)
       
        
        if self.args["pretrain"] == 1:
            pretraining_path = '%s/pretrained_%s_%s_lr_%s_batch_%s_%s_final.pickle' % (self.args["pretrain_path"], self.args["dataset"], self.args["network"], self.args["lr"], self.args["batch_size"], 500)
            with open(pretraining_path, "rb") as f:
                pretrain_w = pickle.load(f)
            model_dict = self.clf.state_dict()
            model_dict.update(pretrain_w)
            self.clf.load_state_dict(model_dict)
            print("loaded pretrained dictionary")

        if nw is not None:
            self.clf.structure_sampler.a_k.requires_grad = False
            self.clf.structure_sampler.b_k.requires_grad = False

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=self.args['l2'])
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
        
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=self.args['l2'])

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        epoch = 0
        accCurrent = 0
        is_saturated = False
        acc_monitor = []
        
        # Pradeep Bajracharya
        best_acc = -1
        patience = 40
        early_stop = 0
        if self.args['es'] == 0:
            print("No Early Stop, default trraining")
            while (accCurrent < self.args['max_accuracy']) and (epoch < n_epoch) and (not is_saturated): 
                print("\n------ EPOCH {} ------".format(epoch))
                print("before", self.clf.structure_sampler.get_variational_params())
                t_start = time.time()
                if gradient_weights is None:
                    accCurrent, lossCurrent, klCurrent, klSc, ELog = self._train(epoch, loader_tr, optimizer, nw=nw)
                    print("AT", self.clf.structure_sampler.get_variational_params())
                else:
                    accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
                t_end = time.time()
                print("after", self.clf.structure_sampler.get_variational_params())
                
                # VALIDATION CHECK Pradeep Bajracharya
                v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=nw)
                print("{}, {}, {}, {}, Training time: {}".format(accCurrent, lossCurrent, v_acc, v_loss, t_end - t_start))
                if v_acc > best_acc:
                    print('Epoch: {}, Model saved, Validation accuracy increased {} -> {}. Saving model ...'.format(epoch, best_acc, v_acc), flush=True)
                    best_acc = v_acc
                    torch.save(self.clf.state_dict(), save_path)
                    # if nw is not None:
                    #     plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")

                if self.args['optimizer'] == 'sgd':
                    lr_sched.step()
                # exit(0)
                # if nw is not None:
                #     plot_network_mask_base(self.clf, save_loc + "/model_epoch"+str(epoch)+".png")
                
                epoch += 1
                # if(self.args['isverbose']):
                #     if epoch % 50 == 0:
                #         print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

                #Stop training if not converging
                if len(acc_monitor) >= self.args['window_size']:

                    is_saturated = self.check_saturation(acc_monitor)
                    del acc_monitor[0]

                log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
                train_logs.append(log_string)
                if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                    self.clf = self.net.apply(weight_reset).to(device=self.device)
                    
                    if self.args['optimizer'] == 'sgd':

                        optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

                    else:
                        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
        elif self.args['es'] == 1:    
            print("Early Stop, patience {}".format(patience))
            while (epoch + 1 < n_epoch) or (patience <= early_stop): 
                print("\n------ EPOCH {} ------".format(epoch))
                # v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=None)
                # print(v_acc, v_loss)
                #
                # print("before", self.clf.structure_sampler.get_variational_params())				
                t_start = time.time()
                if gradient_weights is None:
                    accCurrent, lossCurrent, klCurrent, klSc, ELog = self._train(epoch, loader_tr, optimizer, nw=nw, valid_set=valid_loader)
                    # print("AT", self.clf.structure_sampler.get_variational_params())
                else:
                    accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
                t_end = time.time()
                # print("{}, {}, Training 1 epoch: {}".format(accCurrent, lossCurrent, t_end - t_start)); exit()
                acc_monitor.append(accCurrent)
                # print("after", self.clf.structure_sampler.get_variational_params())
                
                # loader_te = DataLoader(valid_set, shuffle=True, pin_memory=True, batch_size=batch_size)
                # for batch_id, (x,y) in enumerate(valid_loader):     
                #     print(y[-20:-15]); exit(0)
                
                # VALIDATION CHECK Pradeep Bajracharya
                v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=nw, epoch=epoch)
                print("{}, {}, {}, {}, Training time: {}".format(accCurrent, lossCurrent, v_acc, v_loss, t_end - t_start))
                # if rounds:
                #     plot_network_mask_base(self.clf, save_loc + "/model_e"+str(epoch)+".png", figsize=True)
                if v_acc > best_acc:
                    print('Model saved, Validation accuracy: {} -> {}'.format(best_acc, v_acc), flush=True)
                    best_acc = v_acc
                    torch.save(self.clf.state_dict(), save_path)
                    early_stop = 1
                    # if nw is not None:
                    #     plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")
                else:
                    early_stop += 1
                    if patience <= early_stop:
                        print("early stopped")
                        break

                if self.args['optimizer'] == 'sgd':
                    lr_sched.step()
                    
                # if nw is not None:
                #     plot_network_mask_base(self.clf, save_loc + "/model_epoch"+str(epoch)+".png")
                
                epoch += 1
                # if(self.args['isverbose']):
                #     if epoch % 50 == 0:
                #         print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

                #Stop training if not converging
                # if len(acc_monitor) >= self.args['window_size']:

                #     is_saturated = self.check_saturation(acc_monitor)
                #     del acc_monitor[0]
                # if test_set is not None:
                #     tst_acc, tst_loss = self.get_acc_on_set(test_set, shuffle=None, nw=nw)
                #     log_string = 'Epoch:' + str(epoch) + '- Tr. acc:'+str(accCurrent)+'- Tr. loss:'+str(lossCurrent) + '- Tst. acc:'+str(tst_acc)+'- Tst. loss:'+str(tst_loss)
                # else:
                #     log_string = 'Epoch:' + str(epoch) + '- Tr. acc:'+str(accCurrent)+'- Tr. loss:'+str(lossCurrent)
                # train_logs.append(log_string)
                # if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                #     self.clf = self.net.apply(weight_reset).to(device=self.device)
                    
                #     if self.args['optimizer'] == 'sgd':

                #         optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                #         lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

                #     else:
                #         optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
                # if epoch == 6:
                #     exit()
                # self.net.get_param_count(3)
                # print("Param Count:" , self.net.get_param_count())
        # print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), 'Training loss:', str(lossCurrent), 'Training kl:', str(klCurrent),flush=True)

        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf
    
    def train_ssl(self, gradient_weights=None, valid_loader=None, test_loader=None, save_loc="./", save_path="./model.pt", nw=None, rounds=False):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        # v_acc, v_loss = self.get_acc_on_set(valid_set, nw=None)
        # print(v_acc, v_loss); exit()
        def weight_reset(m):
            
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            
            # with open(save_loc + "/train_model_test.pickle", 'rb') as f:
            #     pretrain_w = pickle.load(f)
            # model_dict = self.net.state_dict()
            # model_dict.update(pretrain_w)
            # self.net.load_state_dict(model_dict)

        train_logs = []
        n_epoch = self.args['n_epoch']
        # self.args['isreset'] = False
        self.clf = self.net.to(device=self.device)
        
        if self.args['isreset']:
            # if nw is not None:
            #     self.clf.structure_sampler.reset_parameters()
            self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)

        if nw is not None:
            self.clf.structure_sampler.a_k.requires_grad = False
            self.clf.structure_sampler.b_k.requires_grad = False

        # with open(save_loc + "/train_model_test.pickle", 'rb') as f:
        #     pretrain_w = pickle.load(f)
        # model_dict = self.clf.state_dict()
        # model_dict.update(pretrain_w)
        # self.clf.load_state_dict(model_dict)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=self.args['l2'])
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
        
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=self.args['l2'])

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        epoch = 0
        accCurrent = 0
        is_saturated = False
        acc_monitor = []
        
        # Pradeep Bajracharya
        best_acc = -1
        patience = 30
        early_stop = 0
        if self.args['es'] == 0:
            print("No Early Stop, default trraining")
            while (accCurrent < self.args['max_accuracy']) and (epoch < n_epoch) and (not is_saturated): 
                print("\n------ EPOCH {} ------".format(epoch))
                t_start = time.time()
                if gradient_weights is None:
                    accCurrent, lossCurrent, klCurrent, klSc, ELog = self._train_ssl(epoch, loader_tr, optimizer, nw=nw)
                    # print("AT", self.clf.structure_sampler.get_variational_params())
                else:
                    accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
                t_end = time.time()
                
                # VALIDATION CHECK Pradeep Bajracharya
                v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=nw)
                print("{}, {}, {}, {}, Training time: {}".format(accCurrent, lossCurrent, v_acc, v_loss, t_end - t_start))
                if v_acc > best_acc:
                    print('Epoch: {}, Model saved, Validation accuracy increased {} -> {}. Saving model ...'.format(epoch, best_acc, v_acc), flush=True)
                    best_acc = v_acc
                    torch.save(self.clf.state_dict(), save_path)
                    # if nw is not None:
                    #     plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")

                if self.args['optimizer'] == 'sgd':
                    lr_sched.step()
                # exit(0)
                # if nw is not None:
                #     plot_network_mask_base(self.clf, save_loc + "/model_epoch"+str(epoch)+".png")
                
                epoch += 1
                # if(self.args['isverbose']):
                #     if epoch % 50 == 0:
                #         print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

                #Stop training if not converging
                if len(acc_monitor) >= self.args['window_size']:

                    is_saturated = self.check_saturation(acc_monitor)
                    del acc_monitor[0]

                log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
                train_logs.append(log_string)
                if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                    self.clf = self.net.apply(weight_reset).to(device=self.device)
                    
                    if self.args['optimizer'] == 'sgd':

                        optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

                    else:
                        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
        elif self.args['es'] == 1:    
            print("Early Stop, patience {}".format(patience))
            while (epoch + 1 < n_epoch) or (patience <= early_stop): 
                print("\n------ EPOCH {} ------".format(epoch))
                # v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=None)
                # print(v_acc, v_loss)		
                t_start = time.time()
                if gradient_weights is None:
                    accCurrent, lossCurrent, klCurrent, klSc, ELog = self._train_ssl(epoch, loader_tr, optimizer, nw=nw, valid_set=valid_loader)
                    # print("AT", self.clf.structure_sampler.get_variational_params())
                else:
                    accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
                t_end = time.time()
                # print("{}, {}, Training 1 epoch: {}".format(accCurrent, lossCurrent, t_end - t_start)); exit()
                acc_monitor.append(accCurrent)
                
                # loader_te = DataLoader(valid_set, shuffle=True, pin_memory=True, batch_size=batch_size)
                # for batch_id, (x,y) in enumerate(valid_loader):     
                #     print(y[-20:-15]); exit(0)
                
                # VALIDATION CHECK Pradeep Bajracharya
                v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=nw, epoch=epoch)
                print("{}, {}, {}, {}, Training time: {}".format(accCurrent, lossCurrent, v_acc, v_loss, t_end - t_start))
                # if rounds:
                #     plot_network_mask_base(self.clf, save_loc + "/model_e"+str(epoch)+".png", figsize=True)
                if v_acc > best_acc:
                    print('Model saved, Validation accuracy: {} -> {}'.format(best_acc, v_acc), flush=True)
                    best_acc = v_acc
                    torch.save(self.clf.state_dict(), save_path)
                    early_stop = 1
                    # if nw is not None:
                    #     plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")
                else:
                    early_stop += 1
                    if patience <= early_stop:
                        print("early stopped")
                        break

                if self.args['optimizer'] == 'sgd':
                    lr_sched.step()
                    
                # if nw is not None:
                #     plot_network_mask_base(self.clf, save_loc + "/model_epoch"+str(epoch)+".png")
                
                epoch += 1
              
        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf
       
        
    def train_bbdrop(self, gradient_weights=None, valid_loader=None, test_loader=None, save_loc="./", save_path="./model.pt", nw=None, rounds=False):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        print(self.args)
        # v_acc, v_loss = self.get_acc_on_set(valid_set, nw=None)
        # print(v_acc, v_loss); exit()
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        n_epoch = self.args['n_epoch']
        # self.args['isreset'] = False
        self.clf = self.net.to(device=self.device)
        if self.args['isreset']:
            if nw is not None:
                self.clf.reset_bb()
            self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)


        if self.args["pretrain"] == 1:
            pretraining_path = '%s/pretrained_%s_%s_lr_%s_batch_%s_%s_final.pickle' % (self.args["pretrain_path"], self.args["dataset"], self.args["network"], self.args["lr"], self.args["batch_size"], 500)
            with open(pretraining_path, "rb") as f:
                pretrain_w = pickle.load(f)
            model_dict = self.clf.state_dict()
            model_dict.update(pretrain_w)
            self.clf.load_state_dict(model_dict)
            print("loaded pretrained dictionary")
        
        self.clf.stop_training()

        
        base_params = []
        gate_params = []
        for name, param in self.clf.named_parameters():
            if 'gate' in name:
                gate_params.append(param)
            else:
                base_params.append(param)
                
        if self.args['optimizer'] == 'sgd':
            # optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=self.args['l2'])
            # lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
            optimizer = optim.Adam([
            {'params': gate_params, 'lr': 1e-2},
            {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-4}])
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[int(r * n_epoch) for r in [.5, .8]],
                                                gamma=0.1)
        
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam([
            {'params': gate_params, 'lr': 1e-2},
            {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-4}])
        
            lr_sched = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[int(r * n_epoch) for r in [.5, .8]],
                                                gamma=0.1)

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        epoch = 0
        accCurrent = 0
        is_saturated = False
        acc_monitor = []
        
        # Pradeep Bajracharya
        best_acc = -1
        patience = 30
        early_stop = 0
        if self.args['es'] == 0:
            print("No Early Stop, default trraining")
            while (accCurrent < self.args['max_accuracy']) and (epoch < n_epoch) and (not is_saturated): 
                print("\n------ EPOCH {} ------".format(epoch))
                # print("Before: ")
                # self.clf.get_params()
                t_start = time.time()
                if gradient_weights is None:
                    accCurrent, lossCurrent, klCurrent, klSc, ELog = self._train_bbdrop(epoch, loader_tr, optimizer, nw=nw)
                else:
                    accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
                t_end = time.time()
                
                # VALIDATION CHECK Pradeep Bajracharya
                v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=nw)
                print("{}, {}, {}, {}, Training time: {}".format(accCurrent, lossCurrent, v_acc, v_loss, t_end - t_start))
                if v_acc > best_acc:
                    print('Epoch: {}, Model saved, Validation accuracy increased {} -> {}. Saving model ...'.format(epoch, best_acc, v_acc), flush=True)
                    best_acc = v_acc
                    torch.save(self.clf.state_dict(), save_path)
                    # if nw is not None:
                    #     plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")

                lr_sched.step()
                # exit(0)
                # if nw is not None:
                #     plot_network_mask_base(self.clf, save_loc + "/model_epoch"+str(epoch)+".png")
                
                epoch += 1
                # if(self.args['isverbose']):
                #     if epoch % 50 == 0:
                #         print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

                #Stop training if not converging
                if len(acc_monitor) >= self.args['window_size']:

                    is_saturated = self.check_saturation(acc_monitor)
                    del acc_monitor[0]

                log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
                train_logs.append(log_string)
                if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                    # self.clf = self.net.apply(weight_reset).to(device=self.device)
                    
                    # if self.args['optimizer'] == 'sgd':

                    #     optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    #     lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

                    # else:
                    #     optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
                    
                    self.clf = self.net.apply(weight_reset).to(device=self.device)
                    
                    base_params = []
                    gate_params = []
                    for name, param in self.clf.named_parameters():
                        if 'gate' in name:
                            gate_params.append(param)
                        else:
                            base_params.append(param)
                            
                    if self.args['optimizer'] == 'sgd':
                        optimizer = optim.Adam([
                        {'params': gate_params, 'lr': 1e-2},
                        {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-4}])
                        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(r * n_epoch) for r in [.5, .8]],
                                                            gamma=0.1)
                    
                    elif self.args['optimizer'] == 'adam':
                        optimizer = optim.Adam([
                        {'params': gate_params, 'lr': 1e-2},
                        {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-4}])
                    
                        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(r * n_epoch) for r in [.5, .8]],
                                                            gamma=0.1)
                # print("After: ")
                # self.clf.get_params()
        elif self.args['es'] == 1:    
            print("Early Stop, patience {}".format(patience))
            while (epoch + 1 < n_epoch) or (patience <= early_stop): 
                print("\n------ EPOCH {} ------".format(epoch))
                # v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=None)
                # print(v_acc, v_loss)		
                t_start = time.time()
                if gradient_weights is None:
                    accCurrent, lossCurrent, klCurrent, klSc, ELog = self._train_bbdrop(epoch, loader_tr, optimizer, nw=nw, valid_set=valid_loader)
                else:
                    accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
                t_end = time.time()
                # print("{}, {}, Training 1 epoch: {}".format(accCurrent, lossCurrent, t_end - t_start)); exit()
                acc_monitor.append(accCurrent)
                
                # loader_te = DataLoader(valid_set, shuffle=True, pin_memory=True, batch_size=batch_size)
                # for batch_id, (x,y) in enumerate(valid_loader):     
                #     print(y[-20:-15]); exit(0)
                
                # VALIDATION CHECK Pradeep Bajracharya
                v_acc, v_loss = self.get_acc_on_set(valid_loader, nw=nw, epoch=epoch)
                print("{}, {}, {}, {}, Training time: {}".format(accCurrent, lossCurrent, v_acc, v_loss, t_end - t_start))
                # if rounds:
                #     plot_network_mask_base(self.clf, save_loc + "/model_e"+str(epoch)+".png", figsize=True)
                if v_acc > best_acc:
                    print('Model saved, Validation accuracy: {} -> {}'.format(best_acc, v_acc), flush=True)
                    best_acc = v_acc
                    torch.save(self.clf.state_dict(), save_path)
                    early_stop = 1
                    # if nw is not None:
                    #     plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")
                else:
                    early_stop += 1
                    if patience <= early_stop:
                        print("early stopped")
                        break

                lr_sched.step()
                    
                # if nw is not None:
                #     plot_network_mask_base(self.clf, save_loc + "/model_epoch"+str(epoch)+".png")
                
                epoch += 1
               

        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf
    
    def train_nas(self, gradient_weights=None, valid_loader=None, test_loader=None, save_loc="./", save_path="./model.pt", nw=None, rounds=False, rd=0):
    
        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        switches = []
        for i in range(14):
            switches.append([True for j in range(len(PRIMITIVES))])
        switches_normal = copy.deepcopy(switches)
        switches_reduce = copy.deepcopy(switches)
        # To be moved to args
        num_to_keep = [5, 3, 1]
        num_to_drop = [3, 2, 2]
        if len(self.args['add_width']) == 3:
            add_width = self.args['add_width']
        else:
            add_width = [0, 0, 0]
        if len(self.args['add_layers']) == 3:
            add_layers = self.args['add_layers']
        else:
            add_layers = [0, 6, 12]
        if len(self.args['dropout_rate']) ==3:
            drop_rate = self.args['dropout_rate']
        else:
            drop_rate = [0.0, 0.0, 0.0]
        eps_no_archs = [10, 10, 10]
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        print('Training..')
        # v_acc, v_loss = self.get_acc_on_set(valid_set, nw=None)
        # print(v_acc, v_loss); exit()
        switches_normal = copy.deepcopy(switches)
        switches_reduce = copy.deepcopy(switches)
        for sp in range(len(num_to_keep)):
            print("SP ", sp)
            self.net = Network(self.args['init_channels'] + int(add_width[sp]), self.args['input_channel'], self.args['CIFAR_CLASSES'], self.args['layers'] + int(add_layers[sp]), criterion, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]))
            # print(model)
            # exit(0)
            model = copy.deepcopy(self.net)
            model = nn.DataParallel(model)
            model = model.cuda()
            print(model.module.get_embedding_dim())
            # idx = getentropy(pool_loader, model)
            logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
            network_params = []
        self.model = self.net.to(device=self.device)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))
        network_params = []
        
        for k, v in model.named_parameters():
                if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                    network_params.append(v)       
        optimizer = torch.optim.SGD(
                network_params,
                self.args['learning_rate'],
                momentum=self.args['momentum'],
                weight_decay=self.args['weight_decay'])
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                    lr=self.args['arch_learning_rate'], betas=(0.5, 0.999), weight_decay=self.args['arch_weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.args['epochs']), eta_min=self.args['learning_rate_min'])
        sm_dim = -1
        epochs = self.args['epochs']
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for epoch in range(self.args['epochs']):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = self._train_nas(loader_tr, valid_loader, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=False)
            else:
                model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.module.update_p()                
                train_acc, train_obj = self._train_nas(loader_tr, valid_loader, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True)
            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_acc, valid_obj = self._infer_nas(valid_loader, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
        torch.save(model.state_dict(), os.path.join(save_path, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # print(model)
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()        
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()        
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            print(idxs, switches_normal, i)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = self.get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = self.get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        print("Normal probs", normal_prob)
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = self.get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = self.get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        print("reduce_prob", reduce_prob)
        logging.info('switches_normal = %s', switches_normal)
        self.logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        self.logging_switches(switches_reduce)
        
        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])                
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = self.parse_network(switches_normal, switches_reduce)
            plot(genotype.normal, os.path.join(save_loc, "{}_normalsp{}_rd{}".format(self.args['acq'], sp, rd)))
            plot(genotype.reduce, os.path.join(save_loc, "{}_reducesp{}_rd{}".format(self.args['acq'], sp, rd)))
            print("Plotted first")
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks                
                num_sk = self.check_sk_number(switches_normal)               
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = self.delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = self.keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = self.keep_2_branches(switches_normal, normal_prob)
                    num_sk = self.check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = self.parse_network(switches_normal, switches_reduce)
                logging.info(genotype)              
            plot(genotype.normal, os.path.join(save_loc, "{}_asnormalsp{}_rd{}".format(self.args['acq'], sp, rd)))
            plot(genotype.reduce, os.path.join(save_loc, "{}_asreducesp{}_rd{}".format(self.args['acq'], sp, rd)))
            print("Plotted after skip")
        print("Nparams", arch_param)
      
    def logging_switches(self, switches):
        for i in range(len(switches)):
            ops = []
            for j in range(len(switches[i])):
                if switches[i][j]:
                    ops.append(PRIMITIVES[j])
            logging.info(ops)  
            
    def parse_network(self, switches_normal, switches_reduce):

        def _parse_switches(switches):
            n = 2
            start = 0
            gene = []
            step = 4
            for i in range(step):
                end = start + n
                for j in range(start, end):
                    for k in range(len(switches[j])):
                        if switches[j][k]:
                            gene.append((PRIMITIVES[k], j - start))
                start = end
                n = n + 1
            return gene
        gene_normal = _parse_switches(switches_normal)
        gene_reduce = _parse_switches(switches_reduce)
        
        concat = range(2, 6)
        
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat, 
            reduce=gene_reduce, reduce_concat=concat
        )
        
        return genotype

    def get_min_k(self, input_in, k):
        input = copy.deepcopy(input_in)
        index = []
        for i in range(k):
            idx = np.argmin(input)
            index.append(idx)
            input[idx] = 1
        
        return index
    
    def get_min_k_no_zero(self, w_in, idxs, k):
        w = copy.deepcopy(w_in)
        index = []
        if 0 in idxs:
            zf = True 
        else:
            zf = False
        if zf:
            w = w[1:]
            index.append(0)
            k = k - 1
        for i in range(k):
            idx = np.argmin(w)
            w[idx] = 1
            if zf:
                idx = idx + 1
            index.append(idx)
        return index
      
    def check_sk_number(self,switches):
        count = 0
        for i in range(len(switches)):
            if switches[i][3]:
                count = count + 1
        
        return count

    def delete_min_sk_prob(self,switches_in, switches_bk, probs_in):
        def _get_sk_idx(switches_in, switches_bk, k):
            if not switches_in[k][3]:
                idx = -1
            else:
                idx = 0
                for i in range(3):
                    if switches_bk[k][i]:
                        idx = idx + 1
            return idx
        probs_out = copy.deepcopy(probs_in)
        sk_prob = [1.0 for i in range(len(switches_bk))]
        for i in range(len(switches_in)):
            idx = _get_sk_idx(switches_in, switches_bk, i)
            if not idx == -1:
                sk_prob[i] = probs_out[i][idx]
        d_idx = np.argmin(sk_prob)
        idx = _get_sk_idx(switches_in, switches_bk, d_idx)
        probs_out[d_idx][idx] = 0.0
        
        return probs_out

    def keep_1_on(self,switches_in, probs):
        switches = copy.deepcopy(switches_in)
        for i in range(len(switches)):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches[i][j]:
                    idxs.append(j)
            drop = self.get_min_k_no_zero(probs[i, :], idxs, 2)
            for idx in drop:
                switches[i][idxs[idx]] = False            
        return switches

    def keep_2_branches(self,switches_in, probs):
        switches = copy.deepcopy(switches_in)
        final_prob = [0.0 for i in range(len(switches))]
        for i in range(len(switches)):
            final_prob[i] = max(probs[i])
        keep = [0, 1]
        n = 3
        start = 2
        for i in range(3):
            end = start + n
            tb = final_prob[start:end]
            edge = sorted(range(n), key=lambda x: tb[x])
            keep.append(edge[-1] + start)
            keep.append(edge[-2] + start)
            start = end
            n = n + 1
        for i in range(len(switches)):
            if not i in keep:
                for j in range(len(PRIMITIVES)):
                    switches[i][j] = False  
        return switches   
      
    def _train_nas(self,train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        for step, (input, target, idxs) in enumerate(train_queue):
            model.train()
            n = input.size(0)
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if train_arch:
                # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
                # the training when using PyTorch 0.4 and above. 
                try:
                    input_search, target_search = next(valid_queue_iter)
                except:
                    valid_queue_iter = iter(valid_queue)
                    input_search, target_search = next(valid_queue_iter)
                input_search = input_search.cuda()
                target_search = target_search.cuda(non_blocking=True)
                optimizer_a.zero_grad()
                logits = model(input_search)
                loss_a = criterion(logits, target_search)
                loss_a.backward()
                nn.utils.clip_grad_norm_(model.module.arch_parameters(), self.args.grad_clip)
                optimizer_a.step()

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(network_params, self.args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg
    
    def _infer_nas(self, valid_queue, model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
                loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def train_base(self, gradient_weights=None, valid_set=None, test_set=None, save_loc="./", save_path="./model.pt", nw=None):
    
        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        train_acc = []
        test_acc = []
        best_val_acc = []
        valid_acc = []
        best_tst_acc = []
        
        n_epoch = self.args['n_epoch']
        print("EPOCHS", n_epoch)
        
        if self.args['isreset']:
            self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            param_list = []
            network_list = []
            # param_list.append(self.clf.conv1.parameters())
            for name, param in self.clf.named_parameters():
                if "structure_sampler" in name:
                    # print('ss', name)
                    network_list.append(param)
                else:
                    # print('nn', name)
                    param_list.append(param)
            
            print("Learning Rate", self.args['lr'])
            optimizer = optim.SGD(params=param_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            
            # optimizer_ada = optim.SGD(params=network_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
            # lr_sched_ada = optim.lr_scheduler.CosineAnnealingLR(optimizer_ada, T_max=n_epoch)
            # print(optimizer, optimizer_ada)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        epoch = 1
        accCurrent = 0
        is_saturated = False
        acc_monitor = []
        
        # Pradeep Bajracharya
        best_acc = -1
        bts_acc = -1
        # early_stop = 0
        # patience = 50

        # weight = frange_cycle_sigmoid(0, 1, n_epoch, 4)
        weights = np.linspace(1/15000, 1, 500)
        # print(weights); exit()
        # print(weight); exit()
        # epoch_counter = 5
        while (epoch < n_epoch): 
            # weight = geometric_discount(len(loader_tr.dataset) // batch_size, 501 - epoch)
            # print(weight)
            # if epoch == 10:
            #     exit()
            # else:
            #     epoch += 1
            #     continue
            weight = weights[epoch]
            st_time = time.time()
            # v_acc, v_loss = self.get_acc_on_set(valid_set)
            if gradient_weights is None:
                # if epoch % epoch_counter == 0:
                #     accCurrent, lossCurrent, kls, klSc, eLog = self._train(epoch, loader_tr, optimizer_ada, nw, 1, opt_for="beta")
                accCurrent, lossCurrent, kls, klSc, eLog = self._train(epoch, loader_tr, optimizer, nw, 1)
            else:
                accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
            
            acc_monitor.append(accCurrent)
            train_acc.append(accCurrent)
            
            # VALIDATION CHECK Pradeep Bajracharya
            v_acc, v_loss = self.get_acc_on_set(valid_set)
            ts_acc, ts_loss = self.get_acc_on_set(test_set)
            # print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), 'Training loss:', lossCurrent, 'Training KL:', kls, 'Training KLSc:', klSc, 'Training E_Log:', eLog, 'Valid accuracy:', round(v_acc, 3), 'Test accuracy:', round(ts_acc, 3), "Time Taken:", time.time() - st_time, flush=True)
            if v_acc > best_acc:
                print('Epoch: {}, Model saved, Validation accuracy: {} -> {}'.format(epoch, best_acc, v_acc), flush=True)
                best_acc = v_acc
                torch.save(self.clf.state_dict(), save_path)
                bts_acc, bts_loss = self.get_acc_on_set(test_set)
                if nw == "ada":
                    plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")
            #     early_stop = 0
            #     print("Early stop reset", early_stop)
            # else:
            #     early_stop += 1
            #     print("Early stop count", early_stop)
            #     if early_stop >= patience:
            #         break
            best_tst_acc.append(bts_acc)
            # torch.save(self.clf.state_dict(), save_loc + "/m_current.pt")  
            best_val_acc.append(best_acc)
            valid_acc.append(v_acc)
            test_acc.append(ts_acc)
            if nw == "ada":
                plot_network_mask_base(self.clf, save_loc + "/model_after_epoch"+str(epoch)+".png")
            
            
            if self.args['optimizer'] == 'sgd':
                lr_sched.step()
                # lr_sched_ada
            
            epoch += 1
            if(self.args['isverbose']):
                if epoch % 50 == 0:
                    print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

            #Stop training if not converging
            if len(acc_monitor) >= self.args['window_size']:

                is_saturated = self.check_saturation(acc_monitor)
                del acc_monitor[0]

            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset).to(device=self.device)
                
                if self.args['optimizer'] == 'sgd':

                    # optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    # lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
                    param_list = []
                    network_list = []
                    # param_list.append(self.clf.conv1.parameters())
                    for name, param in self.clf.named_parameters():
                        if "structure_sampler" in name:
                            # print('ss', name)
                            network_list.append(param)
                        else:
                            # print('nn', name)
                            param_list.append(param)
                    optimizer = optim.SGD(params=param_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            
                    optimizer_ada = optim.SGD(params=network_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
                    lr_sched_ada = optim.lr_scheduler.CosineAnnealingLR(optimizer_ada, T_max=n_epoch)

                else:
                    optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)


            

        if self.args['islogs']:
            return self.clf, train_logs, train_acc, valid_acc, test_acc, best_val_acc, best_tst_acc
        else:
            return self.clf, train_acc, valid_acc, test_acc, best_val_acc, best_tst_acc
    
    def train_base_async(self, gradient_weights=None, valid_set=None, test_set=None, save_loc="./", save_path="./model.pt", nw=None):
        
        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')#; exit(0)
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        train_acc = []
        test_acc = []
        best_val_acc = []
        valid_acc = []
        best_tst_acc = []
        
        n_epoch = self.args['n_epoch']
        print("EPOCHS", n_epoch)
        
        if self.args['isreset']:
            self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)

        if self.args['optimizer'] == 'sgd':
            # optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            param_list = []
            network_list = []
            # param_list.append(self.clf.conv1.parameters())
            for name, param in self.clf.named_parameters():
                if "structure_sampler" in name:
                    # print('ss', name)
                    network_list.append(param)
                else:
                    # print('nn', name)
                    param_list.append(param)
            
            print("Learning Rate", self.args['lr'])
            optimizer = optim.SGD(params=param_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            
            optimizer_ada = optim.SGD(params=network_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
            lr_sched_ada = optim.lr_scheduler.CosineAnnealingLR(optimizer_ada, T_max=n_epoch)
            # print(optimizer, optimizer_ada)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        epoch = 1
        accCurrent = 0
        is_saturated = False
        acc_monitor = []
        
        # Pradeep Bajracharya
        best_acc = -1
        bts_acc = -1
        # early_stop = 0
        # patience = 50

        # weight = frange_cycle_sigmoid(0, 1, n_epoch, 4)
        # print(weight); exit()
        # epoch_counter = 5
        while (epoch < n_epoch): 
            # weight = geometric_discount(len(loader_tr.dataset) // batch_size, epoch - 1)
            st_time = time.time()
            # v_acc, v_loss = self.get_acc_on_set(valid_set)
            if gradient_weights is None:
                if epoch < 5:
                # if epoch % epoch_counter == 0:
                    accCurrent, lossCurrent, kls, klSc, eLog = self._train(epoch, loader_tr, optimizer_ada, nw, 1, opt_for="beta")
                accCurrent, lossCurrent, kls, klSc, eLog = self._train(epoch, loader_tr, optimizer, None, 1)
            else:
                accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights, nw)
            
            
            acc_monitor.append(accCurrent)
            train_acc.append(accCurrent)
            
            # VALIDATION CHECK Pradeep Bajracharya
            v_acc, v_loss = self.get_acc_on_set(valid_set)
            ts_acc, ts_loss = self.get_acc_on_set(test_set)
            # print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), 'Training loss:', lossCurrent, 'Training KL:', kls, 'Training KLSc:', klSc, 'Training E_Log:', eLog, 'Valid accuracy:', round(v_acc, 3), 'Test accuracy:', round(ts_acc, 3), "Time Taken:", time.time() - st_time, flush=True)
            if v_acc > best_acc:
                print('Epoch: {}, Model saved, Validation accuracy: {} -> {}'.format(epoch, best_acc, v_acc), flush=True)
                best_acc = v_acc
                torch.save(self.clf.state_dict(), save_path)
                bts_acc, bts_loss = self.get_acc_on_set(test_set)
                if nw == "ada":
                    plot_network_mask_base(self.clf, save_loc + "/model_best_epoch"+str(epoch)+".png")
            #     early_stop = 0
            #     print("Early stop reset", early_stop)
            # else:
            #     early_stop += 1
            #     print("Early stop count", early_stop)
            #     if early_stop >= patience:
            #         break
            best_tst_acc.append(bts_acc)
            # torch.save(self.clf.state_dict(), save_loc + "/m_current.pt")  
            best_val_acc.append(best_acc)
            valid_acc.append(v_acc)
            test_acc.append(ts_acc)
            if nw == "ada":
                plot_network_mask_base(self.clf, save_loc + "/model_after_epoch"+str(epoch)+".png")
            
            
            if self.args['optimizer'] == 'sgd':
                lr_sched.step()
                lr_sched_ada.step()
            
            epoch += 1
            if(self.args['isverbose']):
                if epoch % 50 == 0:
                    print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

            #Stop training if not converging
            if len(acc_monitor) >= self.args['window_size']:

                is_saturated = self.check_saturation(acc_monitor)
                del acc_monitor[0]

            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset).to(device=self.device)
                
                if self.args['optimizer'] == 'sgd':

                    # optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    # lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
                    param_list = []
                    network_list = []
                    # param_list.append(self.clf.conv1.parameters())
                    for name, param in self.clf.named_parameters():
                        if "structure_sampler" in name:
                            # print('ss', name)
                            network_list.append(param)
                        else:
                            # print('nn', name)
                            param_list.append(param)
                    optimizer = optim.SGD(params=param_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            
                    optimizer_ada = optim.SGD(params=network_list, lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
                    lr_sched_ada = optim.lr_scheduler.CosineAnnealingLR(optimizer_ada, T_max=n_epoch)

                else:
                    optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)


            

        if self.args['islogs']:
            return self.clf, train_logs, train_acc, valid_acc, test_acc, best_val_acc, best_tst_acc
        else:
            return self.clf, train_acc, valid_acc, test_acc, best_val_acc, best_tst_acc