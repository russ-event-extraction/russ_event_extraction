import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(1337)

import collections as col

import os
from datetime import datetime
import json

class config(object):

    emb_dim        = 128
    dep_dim        = 128
    pos_dim        = 128
    vocab_size     = 50000
    hidden_dim     = 512
    num_layers     = 1
    num_directions = 2

    batch_size     = 16
    lr             = 0.00025
    decay_rate     = 0.99
    device         = 0
    epochs         = 0
    patience       = 5

conf = config()
USE_CUDA = torch.cuda.is_available()
device = torch.device(conf.device)



def mapping(_list):
    _2id={}
    for i, item in enumerate(_list):
        _2id[item]=i+1

    return _2id


def prepare_data2id(word_seq, dep_seq, pos_seq):
    word_vocab=set()
    dep_vocab=set()
    pos_vocab=set()
    bag_of_words = []

    for w_seq_i, d_seq_i, p_seq_i in zip(word_seq, dep_seq, pos_seq):
        for w_i, d_i, p_i in zip(w_seq_i, d_seq_i, p_seq_i):
            bag_of_words.append(w_i)
            word_vocab.add(w_i)
            dep_vocab.add(d_i)
            pos_vocab.add(p_i)

    freq_words = col.Counter(bag_of_words).items()
    freq_words=sorted(freq_words, key=lambda s:s[-1], reverse=True)
    vocab_size = min(len(freq_words), conf.vocab_size)
    freq_vocab=[]
    for w in freq_words[:vocab_size]:
        freq_vocab.append(w[0])

    vocab_set=set(freq_vocab)
    dep_vocab=list(dep_vocab)
    pos_vocab=list(pos_vocab)

    word2id=mapping(freq_vocab)
    dep2id=mapping(dep_vocab)
    pos2id=mapping(pos_vocab)

    return word2id,     dep2id,        pos2id,     freq_vocab
