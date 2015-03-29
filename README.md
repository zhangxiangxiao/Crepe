# Crepe

This repository contains code in Torch 7 for text classification from character-level using convolution networks. It can be used to reproduce the results in the following article
Xiang Zhang, Yann LeCun. Text Understanding from Scratch. [Arxiv 1502.01710](http://arxiv.org/abs/1502.01710).

## Components

This repository contains the following components:

* data: data preprocessing scripts. It can be used to convert csv format to a Torch 7 binary format that can be used by the training program directly. We used csv format to distribute the datasets in our article. The datasets are available at [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq).
* train: training program.

For more information, please refer to the readme files in each component directory.
