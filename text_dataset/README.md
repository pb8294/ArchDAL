# On the Interdependence between Data Selection and Architecture Optimization in Deep Active Learning

Deep active learning (DAL) studies the optimal selection of labeled data for training deep neural networks (DNNs). While data selection in traditional active learning is mostly optimized for given features, in DNN these features are learned and change with the learning process as well as the choices of DNN architectures. How is the optimal selection of data affected by this change is not well understood in DAL. To shed light on this question, we present the first systematic investigation on: 1) the relative performance of representative modern DAL data selection strategies, as the architecture types and sizes change in the underlying DNN architecture (Focus 1), and 2) the effect of optimizing the DNN architecture of a DNN on DAL (Focus 2). The results suggest that the change in the DNN architecture significantly influences and outweighs the benefits of data selection in DAL. These results cautions the community in generalizing DAL findings obtained on specific architectures, while suggesting the importance to optimize the DNN architecture in order to maximize the effect of active data selection in DAL.

# CODE FOR EXPERIMENT ON TEXT DATASET

This code repository is developed on top of [dal-toolbox github repo](https://github.com/dhuseljic/dal-toolbox) for implementing Deep Active Learning (DAL) framework with different acquisition strategies across BERT, DISTILBERT and ROBERTA for text datasets (AGNEWS, BANKS77, DBPEDIA and QNLI). The code presented here is used as base code for TMLR paper under same name. 

Current Repository is **Version 1** for the code repository and will be constantly updated to increase efficiency of the code. 

# **EXPERIMENTS**

## **ACTIVE LEARNING METHODS**
- Random
- Margin
- Least Confidence
- Entropy
- BALD
- Coreset
- BADGE
<br >

## **NETWORKS USED**
- BERT
- DISTILBERT
- ROBERTA
<br >

All the code used in this experiment are in folder **/experiments/aglae**

## **RUNNING THE EXPERIMENT**
### **NORMAL RUNNING**
```
python al_txt_L.py --config-name=config_file_name
```

The config_file_name for config file which defines the dataset used, the seed, the network as well as data acquisition is defined in /experiments/aglae/config

### **RUNNING WITH PRETRAINED MODEL**
```
python al_txt_L_.py --config-name=config_file_name
```
The config_file_name for config file which defines the dataset used, the seed, the network as well as data acquisition is defined in /experiments/aglae/config

### **PRETRAINING THE MODEL**
To pre-trained model, we use
```
python al_txt_L_pretrain.py --config-name=config_file_name
```

# **REFERENCES**
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018
- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
- Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.

