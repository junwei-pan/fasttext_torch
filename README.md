This is an Torch implementation of fasttext based on A. Joulin's paper [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759).
Author: Junwei Pan
Email: pandevirus@gmail.com

## Requirements
This code is written in Lua and requires [Torch](http://torch.ch/). If you're on Ubuntu, installing Torch in your home directory may look something like: 
```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

This code also require the `nn` package:
```bash
$ luarocks install nn
```

## Usage

First down load the [sentiment analysis data](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) mentioned in Xiang Zhang's paper: [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). We use the ag_news_csv dataset for training and evaluation. 

Then run the following commands to train and evaluate the **fasttext** model:
```bash
$ th main.lua -corpus_train data/ag_news_csv/train.csv -corpus_test data/ag_news_csv/test.csv -dim 10 -minfreq 10 -stream 1 -epochs 5 -suffix 1 -n_classes 4 -n_gram 1 -flag_decay 0 -lr 0.25
```

The trained model can get an accuracy of 90.93% on the g_news_csv dataset using only the unigram word embeddings.

## To be done

1. Support hashtrick
2. Efficiency improvement

## Acknowledgements

This code is based on the [word2vec_torch](https://github.com/kemaswill/word2vec_torch) project, which extends Yoon Kim's [word2vec_torch](https://github.com/yoonkim/word2vec_torch) by implementing the Continuous Bag-of-words Model.
