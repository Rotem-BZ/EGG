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
    _a, word_indexing_dict, _b = create_embedder(method)
    return list(word_indexing_dict.keys())

#########################

class MyIterableDataset(torch.utils.data.IterableDataset, ABC):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, num_letters: int):
        super(MyDataset, self).__init__()
        self.letters = list('abcdefghijklmnopqrstuvwxyz')[:num_letters]
        self.num_letters = num_letters

    def __getitem__(self, item):
        assert isinstance(item, int), "indices must be integers in my dataset"
        item = item % self.num_letters
        return self.letters[item]

    def __len__(self):
        return self.num_letters


class Sender(nn.Module):
    def __init__(self, backbone, tokenizer, backbone_out_dim, num_letters):
        super(Sender, self).__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.fc = nn.Linear(backbone_out_dim, num_letters)

    def forward(self, x, aux_input=None):
        with torch.no_grad():
            encoded_input = self.tokenizer(x, return_tensors='pt')
            x = self.backbone(**encoded_input)
        x = self.fc(x)
        return x


class Receiver(nn.Module):
    def __init__(self, input_size, num_letters):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_size, num_letters)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc(channel_input)
        return x


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    loss_value = 1 if sender_input == receiver_output else 0
    return loss_value, {}


def old_main():
    num_letters = 5
    bert_output_dim = 768

    opts = core.init(params=['--random_seed=7',  # will initialize numpy, torch, and python RNGs
                             '--lr=1e-3',  # sets the learning rate for the selected optimizer
                             '--batch_size=32',
                             '--optimizer=adam',
                             '--no_cuda'])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Hugging Face stuff ##################
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    print("BERT device", bert_model.device)
    # text = "Replace me by any text you'd like."
    # encoded_input = bert_tokenizer(text, return_tensors='pt')
    # output = bert_model(**encoded_input)
    # print(encoded_input)
    # print(output[0].shape)
    # ######################################
    dataset = MyDataset(num_letters=num_letters)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    sender = Sender(backbone=bert_model, tokenizer=bert_tokenizer,
                    backbone_out_dim=bert_output_dim, num_letters=num_letters)
    sender = core.ReinforceWrapper(sender)
    receiver = Receiver(input_size=1, num_letters=num_letters).to(device)
    receiver = core.SymbolReceiverWrapper(receiver, num_letters, agent_input_size=1)
    receiver = core.ReinforceDeterministicWrapper(receiver)

    game = core.SymbolGameReinforce(sender, receiver, loss, sender_entropy_coeff=0.05, receiver_entropy_coeff=0.0)
    optimizer = torch.optim.Adam(game.parameters(), lr=1e-2)  # we can also use a manually set optimizer

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=dataloader,
                           validation_data=dataloader)
    trainer.train(n_epochs=2)
    # train_example = 'a b'
    # print(train_example)
    # encoded_input = bert_tokenizer(train_example, return_tensors='pt')
    # print(encoded_input)
    # print({k:v.device for k,v in encoded_input.items()})
    # encoded_input = {k:v.to(device) for k,v in encoded_input.items()}
    # output = bert_model(**encoded_input)