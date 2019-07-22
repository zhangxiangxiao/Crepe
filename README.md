# Crepe
[![DOI](https://zenodo.org/badge/12639/zhangxiangxiao/Crepe.svg)](http://dx.doi.org/10.5281/zenodo.17308)

This repository contains code in Torch 7 for text classification from character-level using convolutional networks. It can be used to reproduce the results in the following article:

Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

**Note**: An early version of this work entitled “Text Understanding from Scratch” was posted in Feb 2015 as [arXiv:1502.01710](http://arxiv.org/abs/1502.01710). The present paper above has considerably more experimental results and a rewritten introduction.

**Note**: See also the [example implementation in NVIDIA DIGITS](https://github.com/NVIDIA/DIGITS/tree/master/examples/text-classification). Using [CuDNN](https://github.com/soumith/cudnn.torch) is 17 times faster than the code here.

## Components

This repository contains the following components:

* data: data preprocessing scripts. It can be used to convert csv format to a Torch 7 binary format that can be used by the training program directly. We used csv format to distribute the datasets in our article. The datasets are available at [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq).
* train: training program.

For more information, please refer to the readme files in each component directory.

## Example Usage

Here is an example of using our data tools and training programs pipeline to replicate the small convolutional network for DBPedia ontology classification in the article. First, clone the project and download the file `dbpedia_csv.tar.gz` from [our storage in Google Drive](http://goo.gl/JyCnZq) to the `data` directory. Then, uncompress the files and build `t7b` files using our dataset tools.
```sh
$ cd data
$ tar -xvf dbpedia_csv.tar.gz
$ qlua csv2t7b.lua -input dbpedia_csv/train.csv -output train.t7b
$ qlua csv2t7b.lua -input dbpedia_csv/test.csv -output test.t7b
$ cd ..
```

In the commands above, you can replace `qlua` by `luajit` as long as it has an associated torch 7 distribution installed. Now there will be 2 files `train.t7b` and `test.t7b` in the `data` directory. Normally, the second step is to go to the `train` directory and change the configurations listed in `config.lua`, especially for data file location and number of output units in the last linear layer. This last linear layer is important because its number of output units should correspond to the number of classes in your dataset. Luckily for this example on DBPedia ontology dataset the configurations are all set. One just needs to go into the `train` directory and start the training process
```sh
$ cd train
$ qlua main.lua
```

This time we have to use `qlua`, because there is a nice visualization using Qt that is updated for every era. Please make sure packages `qtlua` and `qttorch` are installed in your system and there is a corresponding X to your terminal. To run this example successfully you will also need a NVidia GPU with at least 3GB of memory. Otherwise, you can configure the model in `train/config.lua` for less parameters.

Okay! If you start to find out checkpointing files like `main_EPOCHES_TIME.t7b` and `sequential_EPOCHES_TIME.[t7b|png]` appearing under the `train` directory in several hours or so, it means the program is running without problems. You should probably find some other entertainment for the day. :P

## Issues

It is discovered that the alphabet actually has two hyphen/minus characters ('-'). This issue was present for the results in the papers as well. Since this is probably a minor issue, we will keep the alphabet configurations as is to ensure reproduceability. That said, you are welcome to change the `alphabet` variable in `train/config.lua` to remove it. See issue #4 for more details.

## Why Call It "Crepe"?

It is just a word popping up to my mind pondering for a repository name in Github. It has nothing to do with French cuisine, text processing or convolutional networks. If a connection is really really needed, how about "Convolutional REPresentation of Expressions"?
