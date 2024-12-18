# On the Interdependence between Data Selection and Architecture Optimization in Deep Active Learning

This code repository is developed on top of [DISTIL github repo](https://github.com/decile-team/distil) for implementing Deep Active Learning (DAL) framework with different acquisition strategies across various RESNETs and VGGs for image datasets (MNIST, CIFAR10, FashionMNIST and SVHN). The code presented here is used as base code for TMLR paper under same name. 

Added on top of baseline is implementation for Architecture Optimization methods including [PDARTs](https://link.springer.com/article/10.1007/s11263-020-01396-x), [BBDropout](https://openreview.net/pdf?id=SylU3jC5Y7) and [Depth-Dropout](https://proceedings.neurips.cc/paper_files/paper/2021/file/dfce06801e1a85d6d06f1fdd4475dacd-Paper.pdf)

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
- RESNET (18, 34, 50)
- VGG (11, 16, 19)
- BBDROPOUT
- DEPTH-DROPOUT
- PDARTS
<br >

## **1. EXPERIMENTS ON RESNETs, VGGs, FIXED CNN NET, DEPTH-DROPOUT, BBDROPOUT AND PDARTs**
### **1.1.FOR RESNETS, VGGs, FIXED CNN MODEL, DEPTH-DROPOUT**
```
python3 main.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --es $es --exp $exp --run $run --optimizer $optimizer --truncation $truncation
```

**dataset** = MNIST or SVHN or FashionMNIST or CIFAR10  
**strategy** = random, entropy, least_confidence, bald, badge, coreset, margin  
**network** = resnet18 or resnet34 or resnet50 or vgg11 or vgg16 or vgg19 or adacnn or simplecnn 
**aug** = 0 for no data augmentation and 1 for data augmentation  
**seed** = set the seed value  
**budget** = Acquisition size  
**es** = 1 to use validation data for early stop and 0 to not do so  
**exp** = Name for experiment folder
**run** = Version of experiment under 'exp' folder
**optimizer** = Choice of optimizer
**load** = To load the model preinitialized
> IMPORTANT  
> Use **truncation** = Number of layers for **Fixed CNN network**  
> Use **truncation** = Number of layers for overcomplete network to infer depth **Fixed Depth-Dropout network**
<br ><br >

### **1.1.FOR BBDropout**
```
python3 main_bbdropout.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --es $es --exp $exp --run $run --optimizer $optimizer --truncation $truncation
```
**truncation** = Number of layers for overcomplete network to infer depth

<br >

### **1.1.FOR PDARTs**
```
python3 main_nas.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --es $es --exp $exp --run $run --optimizer $optimizer
```
<br >

## **2. EXPERIMENTS TO USE PRETRAINED MODEL**
For all the following experiments the path of pretrained network needs to be updated in respective code
<br >

### **2.1. TO USE PRETRAINED MODEL FOR RESNETS, VGGs, SIMPLECNN, DEPTHDROPOUT**
```
python3 main_pretrain.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --pretrain 1 --es $es --exp $exp --run $run --optimizer $optimizer --truncation $truncation
```

### **2.2. TO USE PRETRAINED MODEL FOR BBDROPOUT**
```
python3 main_bbdropout_pretrain.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --pretrain 1 --es $es --exp $exp --run $run --optimizer $optimizer --truncation $truncation
```
<br >

## **3. EXPERIMENTS TO PRETRAIN NETWORK IN UNSUPERVISED METHOD**
### **3.1. FOR RESNETS, VGGS, DEPTH-DROPOUT, FIXED CNNs**
```
python3 main_unsupervised.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --pretrain $pretrain --es $es --exp $exp --run $run --optimizer $optimizer
```
<br >

### **3.2. FOR BBDROPOUT**
```
python3 main_bbdropout_pretrain.py --dataset $dataset --strategy $acq --network $network --aug $aug --load $load --seed $seed --budget $budget --pretrain 100 --es $es --exp $exp --run $run --optimizer $optimizer
```
To pretrain BBDROPOUT model and save the model with unsupervised method set **pretrain** to **100**
<br >
<br >
## **4. TO PRETRAIN IN SUPERVISED METHOD**
To generate pretrained model for all networks in supervised manner, we need to run experiments in **section 1** by replacing **dt.train** with **dt.train_one** and use the trained model in Section 2.
<br ><br >

# **REFERENCES**
- Chen, Xin, et al. "Progressive darts: Bridging the optimization gap for nas in the wild." International Journal of Computer Vision 129 (2021): 638-655.
- KC, Kishan, Rui Li, and MohammadMahdi Gilany. "Joint inference for neural network depth and dropout regularization." Advances in neural information processing systems 34 (2021): 26622-26634.
- Lee, Juho, et al. "Adaptive network sparsification with dependent variational beta-bernoulli dropout." arXiv preprint arXiv:1805.10896 (2018).

