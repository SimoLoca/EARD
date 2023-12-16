# EARD
Master Thesis 2022-2023 about Egocentric Activity Recognition and Detection using Deep Learning

Our code is built upon the codebase from [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [Detectron2](https://github.com/facebookresearch/detectron2).

The proposed solution can achieve performance comparable to current state-of-the-art models, such as [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [TriDet](https://github.com/dingfengshi/TriDet), using fewer parameters and with lower latencies (see section [Comparison](#comparison)).

## Requirements

- Linux (Ubuntu 22.04)
- Python 3.5+ (3.10.6)
- PyTorch 1.11+
- TensorBoard
- CUDA 11.0+
- GCC 4.9+
- 1.11 <= Numpy <= 1.23
- PyYaml
- Pandas
- h5py
- joblib

```shell
pip install  -r requirements.txt
```

## Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.

## Introduction

## Code Overview
The structure of this code repo is heavily inspired by Detectron2. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and pre/post-processing.

To quickly get start with the model architecture, you can focus mainly on the following files:

- `libs/modeling/blocks.py`
- `libs/modeling/backbones.py`
- `libs/modeling/meta_archs.py`


# Data Preparation

- We adopt the feature for **Epic-Kitchen** datasets from ActionFormer repository ([see here](https://github.com/happyharrycn/actionformer_release)). To use these features, please download them from their link and unpack them into the `./data` folder.

* The file includes SlowFast features as well as action annotations in json format (similar to ActivityNet annotation format).

**Details**: The features are extracted from the SlowFast model pretrained on the training set of EPIC Kitchens 100 (action classification) using clips of `32 frames` at a frame rate of `30 fps` and a stride of `16 frames`. This gives one feature vector per `16/30 ~= 0.5333` seconds.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───epic_kitchens/
│    │	 └───annotations
│    │	 └───features   
│    └───...
|
└───libs
│
│   ...
```

## Training
* On EPIC Kitchens, we train separate models for nouns and verbs.
* To train our model on verbs with SlowFast features, use
```shell
python ./train.py ./configs/epic_slowfast_verb.yaml --output reproduce
```
* To train our model on nouns with SlowFast features, use
```shell
python ./train.py ./configs/epic_slowfast_noun.yaml --output reproduce
```

## Evaluate
* Evaluate the trained model for verbs. The expected average mAP should be around 24.66(%).
```shell
python ./eval.py ./configs/epic_slowfast_verb.yaml ./ckpt/epic_slowfast_verb_reproduce
```
* Evaluate the trained model for nouns. The expected average mAP should be around 22.41(%).
```shell
python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_reproduce
```

The results (mAP at tIoUs) should be

| Method              |  0.1  |  0.2  |  0.3  |  0.4  |  0.5  |  Avg  |
|---------------------|-------|-------|-------|-------|-------|-------|
| Model (verb) | 28.01 | 26.93 | 25.57 | 23.45 | 19.31 | 24.66 |
| Model (noun) | 25.94 | 24.91 | 23.26 | 20.54 | 17.42 | 22.41 |

## Comparison
We compared the proposed solution with current sota models: [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [TriDet](https://github.com/dingfengshi/TriDet).

The results were obtained using the same hardware (Nvidia GeForce GTX 1650) testing the whole models during inference on an EPIC-KITCHEN's video.

| Method  (verbs)            |  GMACs  |  Parameters (M)  |  Latency (ms)  | 
|---------------------|-------|-------|-------|
| ActionFormer | 46.6 | 29.76 | 502.36 |
| TriDet | 48.08 | 18.59 | 636.28 |
| *Ours* | 38.74 | 23.16 | 367.43 |

| Method  (nouns)            |  GMACs  |  Parameters (M)  |  Latency (ms)  | 
|---------------------|-------|-------|-------|
| ActionFormer | 48.18 | 30.07 | 515.00 |
| TriDet | 52.49 | 19.50 | 879.51 |
| *Ours* | 40.30 | 23.47 | 383.94 |
