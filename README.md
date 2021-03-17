# Vis-Causal
Code and dataset for the paper "Learning Contextual Causality from Time-consecutive Images".

The paper focuses on contextual causaltiy, referring to the cases where the causality exists only in a certain context. In summary, we create a high quality dataset named "Vis-Causal" containing the contextual causal relationships mining from visual signals (time-consecutive images here).
We also evaluate a proposed VCC model on this dataset to prove its ability of mining contextual causality knowledge.

## Dataset
The dataset includes two parts:

* Download images [HERE](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155160328_link_cuhk_edu_hk/EofV9h11SnZKhR8NtHX-o4YBTooSa5QxHiuQPg9bG1_eaQ?e=0EYLe2).
* Annotations for contextual causality knowledge is in `Contextual_dataset.json`. This file also includes all required data for implementing our model.
* \[*Optional*\]: `video2id.json` provides video IDs in the ActivityNet which we cropped the time-consecutive images from.

## Data Preparing
* Please link the dataset `Contextual_dataset.json` in `"dataset_path"` of `Main.py`. 
* Please download [GloVe](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) word representation and rename it to `glove.txt` under this folder. 

## Train and Evaluate the VCC model
As in the paper, we provide five models for the task:  

- NoContext
- NoAttention
- ResNetAsContext
- Full-Model
- GPT-2

Examples of training and evaluation command (in Full-Model mode):  
```
python Main.py --gpu 0 --model Full-Model --lr 0.0001
```
where `model` is the model name, `lr` is the learning rate.

Note that we leave out the resnet representation for each image in current dataset version due to the upload size limitation.
If you want to evaluate the model in `ResNetAsContext` mode, please extract features from [ResNet-152](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

Welcome to contact us (ythuo at cse.cuhk.edu.hk) for any questions!
