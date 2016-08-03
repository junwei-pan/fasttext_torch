-- Implementation of fasttext(https://arxiv.org/abs/1607.01759) using Torch
-- Author: Junwei Pan, Yahoo Inc.
-- Date: Aug 2, 2016

require("io")
require("os")
require("paths")
require("torch")
dofile("fasttext.lua")

-- Default configuration
config = {}
config.corpus_train = "data/ag_news_csv/train.csv" -- train data
config.corpus_test = "data/ag_news_csv/test.csv" -- test data
config.dim = 100 -- dimensionality of word embeddings
config.minfreq = 10 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.decay = 0 -- whether to decay the learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 3 -- number of epochs to train
config.stream = 1 -- 1 = stream from hard drive 0 = copy to memory first
config.n_classes = 4 -- number of classification classes
config.n_gram = 1 -- n_gram: 1 for unigram, 2 for bigram, 3 for trigram
config.suffix = "" -- suffix for model id
config.title = 1  -- whether to use title
config.description = 1  -- whether to use description


-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus_train", config.corpus_train)
cmd:option("-corpus_test", config.corpus_test)
cmd:option("-minfreq", config.minfreq)
cmd:option("-dim", config.dim)
cmd:option("-lr", config.lr)
cmd:option("-decay", config.decay)
cmd:option("-min_lr", config.min_lr)
cmd:option("-epochs", config.epochs)
cmd:option("-stream", config.stream)
cmd:option("-suffix", config.suffix)
cmd:option("-n_classes", config.n_classes)
cmd:option("-n_gram", config.n_gram)
cmd:option("-title", config.title)
cmd:option("-description", config.description)
params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

for i,j in pairs(config) do
    print(i..": "..j)
end
-- Train model
m = FastText(config)
m:build_vocab(config.corpus_train)
--m:build_table()

for k = 1, config.epochs do
    m.lr = config.lr -- reset learning rate at each epoch
    m:train_model(config.corpus_train)
    m:test_model(config.corpus_test)
end

m:print_sim_words({"one", "second", "city", "man", "china"}, 10)

-- Save the model as well as the word vectors
path_model = 'model/model_'  .. config.suffix
path_vector = 'model/vector_' .. config.suffix
m:save_model(path_model)
m:save_vector(path_vector)
