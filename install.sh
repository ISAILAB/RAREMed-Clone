#!/bin/bash

conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

conda install pandas dill tqdm scikit-learn tensorboard matplotlib jupyter notebook
pip install rdkit