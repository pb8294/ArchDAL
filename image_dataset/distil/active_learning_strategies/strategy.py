import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import time
class Strategy:
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): #
        
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.model = net
        self.target_classes = nclasses
        self.args = args
        self.nw = args['nw']
        # self.nw = nw
        
        if 'batch_size' not in args:
            args['batch_size'] = 1
        
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']
            
        if 'loss' not in args:
            self.loss = F.cross_entropy
        else:
            self.loss = args['loss']

    def select(self, budget):
        pass

    def update_data(self, labeled_dataset, unlabeled_dataset): #
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        
    def update_queries(self, query_dataset):
        self.query_dataset= query_dataset

    def update_privates(self, private_dataset):
        self.private_dataset= private_dataset

    def update_model(self, clf):
        self.model = clf

    def predict(self, to_predict_dataset):
    
        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold class predictions
        P = torch.zeros(len(to_predict_dataset)).long().to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        evaluated_instances = 0
        
        with torch.no_grad():
            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Predict the most likely class
                elements_to_predict = elements_to_predict.to(self.device)
                out = self.model(elements_to_predict)
                if self.nw == 'ada1':
                    out = out.mean(0)
                
                
                pred = out.max(1)[1]
                
                # Insert the calculated batch of predictions into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + elements_to_predict.shape[0]
                P[start_slice:end_slice] = pred
                evaluated_instances = end_slice
                
        return P

    def predict_prob(self, to_predict_dataset):

        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        evaluated_instances = 0
        
        with torch.no_grad():
            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Calculate softmax (probabilities) of predictions
                elements_to_predict = elements_to_predict.to(self.device)
                out = self.model(elements_to_predict)
                if self.nw == 'ada1':
                    out = out.mean(0)
                
                
                pred = F.softmax(out, dim=1)
                
                # Insert the calculated batch of probabilities into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + elements_to_predict.shape[0]
                probs[start_slice:end_slice] = pred
                evaluated_instances = end_slice

        return probs        

    def predict_prob_dropout(self, to_predict_dataset, n_drop):

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                evaluated_instances = 0
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                    if self.nw == 'ada1':
                        out = out.mean(0)
                    
                    
                    pred = F.softmax(out, dim=1)
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[start_slice:end_slice] += pred
                    evaluated_instances = end_slice

        # Divide through by n_drop to get average prob.
        probs /= n_drop

        return probs         

    def predict_prob_dropout_split(self, to_predict_dataset, n_drop):
        
        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                evaluated_instances = 0
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                    if self.nw == 'ada1':
                        out = out.mean(0)
                    
                    
                    pred = F.softmax(out, dim=1)
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice

        return probs 

    def get_embedding(self, to_predict_dataset):
        
        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold embeddings
        embedding = torch.zeros([len(to_predict_dataset), self.model.get_embedding_dim()]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        evaluated_instances = 0
        
        with torch.no_grad():

            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Calculate softmax (probabilities) of predictions
                elements_to_predict = elements_to_predict.to(self.device)
                out, l1 = self.model(elements_to_predict, last=True)
                if self.nw == 'ada1':
                    out = out.mean(0)
                    l1 = l1.mean(0)
                
                
                # Insert the calculated batch of probabilities into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + elements_to_predict.shape[0]
                embedding[start_slice:end_slice] = l1
                evaluated_instances = end_slice

        return embedding

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self, dataset, predict_labels, grad_embedding_type="bias_linear"):
        embDim = self.model.get_embedding_dim()
        self.model = self.model.to(self.device)
        
        # Create the tensor to return depending on the grad_embedding_type, which can have bias only, 
        # linear only, or bias and linear
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(dataset), self.target_classes]).to(self.device)
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(dataset), embDim * self.target_classes]).to(self.device)
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros([len(dataset), (embDim + 1) * self.target_classes]).to(self.device)
        else:
            raise ValueError("Grad embedding type not supported: Pick one of 'bias', 'linear', or 'bias_linear'")
          
        # Create a dataloader object to load the dataset
        dataloader = DataLoader(dataset, batch_size = self.args['batch_size'], shuffle = False)  
          
        evaluated_instances = 0
        
        # If labels need to be predicted, then do so. Calculate output as normal.
        if predict_labels:
            for batch_idx, unlabeled_data_batch in enumerate(dataloader):
                start_slice = evaluated_instances
                end_slice = start_slice + unlabeled_data_batch.shape[0]
                
                inputs = unlabeled_data_batch.to(self.device, non_blocking=True)
                out, l1 = self.model(inputs, last=True, freeze=True)
                # print(out.shape, l1.shape)
                # print("here"); exit(0)
                if self.nw == 'ada1':
                    out = out.mean(0)
                    l1 = l1.mean(0)
                targets = out.max(1)[1]
                
                # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
                loss = self.loss(out, targets, reduction="sum")
                l0_grads = torch.autograd.grad(loss, out)[0]

                # Calculate the linear layer gradients as well if needed
                if grad_embedding_type != "bias":
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.target_classes)
                # Populate embedding tensor according to the supplied argument.
                if grad_embedding_type == "bias":                
                    grad_embedding[start_slice:end_slice] = l0_grads
                elif grad_embedding_type == "linear":
                    grad_embedding[start_slice:end_slice] = l1_grads
                else:
                    grad_embedding[start_slice:end_slice] = torch.cat([l0_grads, l1_grads], dim=1) 
            
                evaluated_instances = end_slice
            
                # Empty the cache as the gradient embeddings could be very large
                torch.cuda.empty_cache()
        else:
        
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                start_slice = evaluated_instances
                end_slice = start_slice + inputs.shape[0]
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                out, l1 = self.model(inputs, last=True, freeze=True)
                if self.nw == 'ada1':
                    out = out.mean(0)
                    l1 = l1.mean(0)
            
                # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
                loss = self.loss(out, targets, reduction="sum")
                l0_grads = torch.autograd.grad(loss, out)[0]

                # Calculate the linear layer gradients as well if needed
                if grad_embedding_type != "bias":
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.target_classes)

                # Populate embedding tensor according to the supplied argument.
                if grad_embedding_type == "bias":                
                    grad_embedding[start_slice:end_slice] = l0_grads
                elif grad_embedding_type == "linear":
                    grad_embedding[start_slice:end_slice] = l1_grads
                else:
                    grad_embedding[start_slice:end_slice] = torch.cat([l0_grads, l1_grads], dim=1) 
            
                evaluated_instances = end_slice
            
                # Empty the cache as the gradient embeddings could be very large
                torch.cuda.empty_cache()
        
        # Return final gradient embedding
        # print(grad_embedding.shape); exit(0)
        return grad_embedding

    def feature_extraction(self, inp, layer_name):
        feature = {}
        model = self.model
        def get_features(name):
            def hook(model, inp, output):
                feature[name] = output.detach()
            return hook
        for name, layer in self.model._modules.items():
            if name == layer_name:
                layer.register_forward_hook(get_features(layer_name))
        output = self.model(inp)
        return torch.squeeze(feature[layer_name])

    def get_feature_embedding(self, dataset, unlabeled, layer_name='avgpool'):
        dataloader = DataLoader(dataset, batch_size = self.args['batch_size'], shuffle = False)
        features = []
        if(unlabeled):
            for batch_idx, inputs in enumerate(dataloader):
                inputs = inputs.to(self.device)
                batch_features = self.feature_extraction(inputs, layer_name)
                features.append(batch_features)
        else:
            for batch_idx, (inputs,_) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                batch_features = self.feature_extraction(inputs, layer_name)
                features.append(batch_features)
        return torch.vstack(features)
    
    '''
    FUNCTION DEFINITION FOR CALCULATING EGL WITH RESPECT TO THE PARAMETERS OF THE NETWORK INFERENCE
    - Looped through all the labels (10 labels --> 0 to 9) for each data and calculated respective loss
    - Backpropagated loss for each label for a single forward pass to calculate/get the gradients
    '''
    def egl_hyperparameter(self, dataset, llabel, weight = 1, data_save_path = "./", grads_asp=0):
        data_loader = DataLoader(dataset, batch_size = 1, shuffle = False)  
        criterion = nn.CrossEntropyLoss()
        criterion.reduction = 'mean'
        self.model.train()
        all_grads = []
        g_norm = []
        all_least_conf = []
        all_entropy = []
        print("Current pool size: ", len(data_loader.dataset))
        th = None
        setflag = 0
        for i, (data, labels) in enumerate(data_loader):
            # print("Data number: {}".format(i + 1))
            single_st = time.time()
            data = data.float().to(self.device)
            labels = labels.long().to(self.device)
            # print(labels.shape)
            # print(labels); exit(0)
            ''' kl_beta_kum has same value regardless of the datasize which is good as the kl is of network parameters
                and not related to data '''
            output = self.model(data)
            logits = torch.softmax(output.mean(0), dim=1)
            # if th is None:
            #     th = 3#max(ths)
            
            # if setflag == 0:
            #     print("Max threshold for gradient lenght: {}".format(th))
            #     setflag = 1
            # print(logits.shape, (-logits * torch.log(logits + 1e-30))); exit(0)
            # if cross_check_acqs:
            #     with torch.no_grad():
            #         # all_least_conf.append(torch.max(logits).detach().cpu())    
            #         all_entropy.append(torch.sum(-logits * torch.log(logits + 1e-30)).detach().cpu())
            
            # weight_grads = 0.0
            weight_grads_a = torch.zeros((20)).cuda()
            weight_grads_b = torch.zeros((20)).cuda()
            l_st = time.time()
            for c in llabel:
                l_st = time.time()
                labels = torch.ones(1).to(self.device) * c
                loss, kls, kl_scale, e_total= self.model.estimate_ELBO(criterion, output, labels, len(data_loader.dataset), kl_weight=weight)
                loss.backward(retain_graph=True)
                w_a = self.model.structure_sampler.a_variational.grad.flatten()#[:th]
                w_b = self.model.structure_sampler.b_variational.grad.flatten()#[:th]
                weight_grads_a += w_a * logits[:, c]
                weight_grads_b += w_b * logits[:, c]
                # weight_grads += (torch.sum(torch.abs(w_a)) + torch.sum(torch.abs(w_b))) * logits[:, c]
                del w_a, w_b
                l_et = time.time()
                self.model.structure_sampler.a_k.grad.zero_()
                self.model.structure_sampler.b_k.grad.zero_()
            # print(weight_grads_a, weight_grads_b)
            # g_norm.append(weight_grads.detach().cpu())
            if grads_asp == 0:
                all_grads.append(torch.cat((weight_grads_a, weight_grads_b), 0).detach().cpu().view(1, -1))
            elif grads_asp == 1:
                all_grads.append(weight_grads_a.detach().cpu().view(1, -1))
            elif grads_asp == 2:
                # print(weight_grads_b.shape, weight_grads_b.detach().cpu().view(1, -1).shape); exit(0)
                all_grads.append(weight_grads_b.detach().cpu().view(1, -1))
                
            # all_grads.append(weight_grads.detach().cpu())
            # if j == 10:
            #     exit(0)
            l_et = time.time()
            single_et = time.time()
            del weight_grads_a, weight_grads_b
        with open(data_save_path + "egl_timing.txt", 'a') as f:
                f.write(f"Single time: {single_et - single_st}, Label time: {l_et - l_st}")
        # if cross_check_acqs:
        #     # all_least_conf = torch.stack(all_least_conf)
        #     all_entropy = torch.stack(all_entropy)

        # g_norm = torch.cat(g_norm) 
        all_grads = torch.cat(all_grads, 0)
        # all_grads_new = torch.hstack((all_grads[:, :3], all_grads[:, 20:23]))
        g_norm = torch.linalg.norm(all_grads, dim=1)
        
        return all_grads.numpy(), g_norm.cpu().numpy()