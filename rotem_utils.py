from abc import ABC

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import egg.core as core
from transformers import BertTokenizer, BertModel

import numpy as np
import pickle
import math
import os
from tqdm import tqdm
from itertools import combinations
import nltk
from nltk.corpus import words, wordnet as wn


def initiate_GloVe(save_data=True):
    word_indexing_dict = dict()
    embeddings = []
    with open('glove.6B.100d.txt', 'rt', encoding="utf-8") as fi:
        full_content = fi.read().strip().split('\n')
    for i in tqdm(range(len(full_content))):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        word_indexing_dict[i_word] = i
        embeddings.append(torch.tensor(i_embeddings))

    embeddings = torch.stack(embeddings)
    unk_vector = embeddings.mean(dim=0, keepdim=True)
    embeddings = torch.cat([embeddings, unk_vector], dim=0)
    glove_data = (embeddings, word_indexing_dict)
    if save_data:
        with open("saved_glove_embeddings.pkl", "wb") as write_file:
            pickle.dump(glove_data, write_file)
    return glove_data


def create_embedder(method: str):
    if method == 'GloVe':
        if os.path.isfile("saved_glove_embeddings.pkl"):
            with open("saved_glove_embeddings.pkl", "rb") as read_file:
                embeddings, word_indexing_dict = pickle.load(read_file)
        else:
            embeddings, word_indexing_dict = initiate_GloVe()
    else:
        raise ValueError("currently, only GloVe embedding is allowed")

    index_word_dict = {v:k for k,v in word_indexing_dict.items()}

    return embeddings, word_indexing_dict, index_word_dict


def get_vocabulary(method: str):
    if method == 'nltk-words':
        try:
            x = words.words()
        except LookupError:
            nltk.download('words')
            x = words.words()
        return x

    if method == 'wordnet-nouns':
        try:
            x = wn.all_synsets('n')
        except LookupError:
            nltk.download('wordnet')
            x = wn.all_synsets('n')
        return list(map(lambda s: s.name().partition('.')[0], x))

    _a, word_indexing_dict, _b = create_embedder(method)
    return list(word_indexing_dict.keys())


def get_embedder_with_vocab(embed_method: str = 'GloVe', vocab_method: str = 'wordnet-nouns'):
    embeddings, word_indexing_dict, index_word_dict = create_embedder(embed_method)
    all_vocab = get_vocabulary(vocab_method)
    vocab = [v for v in all_vocab if v in word_indexing_dict]
    print(f"{100 * len(vocab) / len(all_vocab)}% of the vocabulary \'{vocab_method}\' can be used with"
          f" \'{embed_method}\'embedding. Total of {len(vocab)} words")
    vocab_indices = [word_indexing_dict[word] for word in vocab]
    embeddings = embeddings[vocab_indices + [-1]]
    word_indexing_dict = {v:i for i,v in enumerate(vocab)}
    index_word_dict = {i: v for i, v in enumerate(vocab)}
    return embeddings, word_indexing_dict, index_word_dict, vocab


class EmbeddingAgent(nn.Module):
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'GloVe',
                 vocab_method: str = 'wordnet-nouns'):
        super(EmbeddingAgent, self).__init__()
        self.gca = good_cards_amount
        self.tca = total_cards_amount
        self.embeddings, self.word_index_dict, self.index_word_dict, _ = get_embedder_with_vocab(embedding_method,
                                                                                                 vocab_method)

    def embedder(self, word: str):
        return self.embeddings[self.word_index_dict.get(word, -1)]


def try_ce_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    loss = F.binary_cross_entropy(receiver_output, sender_input.view(-1, 784), reduction='none').mean(dim=1)
    return loss, {}

#########################

