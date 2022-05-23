import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import pandas as pd
import pickle
import time
import os
from os.path import join, isdir, isfile
from tqdm import tqdm

import nltk
from nltk.corpus import words, wordnet as wn
import gensim.downloader as api
from transformers import BertTokenizer, BertModel

# global variable - name of embedding savings directory
embedding_saves_dir = "embeddings_saved"
glove_embeddings_file = "saved_glove_embeddings.pkl"
word2vec_embeddings_file = "saved_w2v_embeddings.pkl"


def load_saved_embedding(file_name):
    if not isdir(embedding_saves_dir):  # no directory 'embeddings_saved'
        os.mkdir(embedding_saves_dir)
        return None

    if not isfile(path := join(embedding_saves_dir, file_name)):  # no file file_name in the directory
        return None

    with open(path, "rb") as read_file:
        x = pickle.load(read_file)
    return x


def dump_embeddings(file_name, obj):
    assert isdir(embedding_saves_dir)
    with open(join(embedding_saves_dir, file_name), "wb") as write_file:
        pickle.dump(obj, write_file)


def user_choice(def_val, text_param: str, options: list = None, demand_int: bool = False, full_text: bool = False):
    assert demand_int or options
    extra_text = 'choose an integer' if demand_int else 'out of ' + ' or '.join(map(lambda x: f"\'{x}\'", options))
    while True:
        inp = input(text_param if full_text else f"What is the {text_param}? (default: {def_val}) \n ({extra_text}) \n")
        if inp == 'a':
            return def_val
        if demand_int:
            if not inp.isdigit():
                print("please choose an integer. \n")
                continue
            return int(inp)
        if inp not in options:
            print("this input is not an option. choose out of: \n", options)
            continue
        return inp


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
        dump_embeddings(glove_embeddings_file, glove_data)
    return glove_data


def initiate_word2vec(save_data=True):
    gensim_model = api.load('word2vec-google-news-300')
    word_indexing_dict = gensim_model.key_to_index
    embeddings = torch.from_numpy(gensim_model.vectors)
    unk_vector = embeddings.mean(dim=0, keepdim=True)
    embeddings = torch.cat([embeddings, unk_vector], dim=0)
    w2v_data = (embeddings, word_indexing_dict)
    if save_data:
        dump_embeddings(word2vec_embeddings_file, w2v_data)
    return w2v_data


def initiate_bert(save_data=True):
    """ some of the code taken from
    https://github.com/arushiprakash/MachineLearning/blob/main/BERT%20Word%20Embeddings.ipynb
    """
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # TODO: implement everything
    raise NotImplementedError


def create_embedder(method: str):
    if method == 'GloVe':
        x = load_saved_embedding(glove_embeddings_file)
        if x is None:
            embeddings, word_indexing_dict = initiate_GloVe()
        else:
            embeddings, word_indexing_dict = x
    elif method == 'word2vec':
        x = load_saved_embedding(word2vec_embeddings_file)
        if x is None:
            embeddings, word_indexing_dict = initiate_word2vec()
        else:
            embeddings, word_indexing_dict = x
    else:
        raise ValueError("currently, only GloVe and word2vec embeddings are allowed")

    index_word_dict = {v: k for k, v in word_indexing_dict.items()}

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

    if method == 'nouns-kaggle':
        # taken from https://www.kaggle.com/datasets/leite0407/list-of-nouns
        return ['atm'] + np.char.lower(pd.read_csv('nounlist.csv').to_numpy().reshape(-1).astype(str)).tolist()

    if method == 'codenames-words':
        return [
            'vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'table', 'lead', 'crown',
            'bomb', 'bug', 'pipe', 'roulette', 'australia', 'play', 'cloak', 'piano', 'beijing', 'bison',
            'boot', 'cap', 'car', 'change', 'circle', 'cliff', 'conductor', 'cricket', 'death', 'diamond',
            'figure', 'gas', 'germany', 'india', 'jupiter', 'kid', 'king', 'lemon', 'litter', 'nut',
            'phoenix', 'racket', 'row', 'scientist', 'shark', 'stream', 'swing', 'unicorn', 'witch', 'worm',
            'pistol', 'saturn', 'rock', 'superhero', 'mug', 'fighter', 'embassy', 'cell', 'state', 'beach',
            'capital', 'post', 'cast', 'soul', 'tower', 'green', 'plot', 'string', 'kangaroo', 'lawyer', 'fire',
            'robot', 'mammoth', 'hole', 'spider', 'bill', 'ivory', 'giant', 'bar', 'ray', 'drill', 'staff',
            'greece', 'press', 'pitch', 'nurse', 'contract', 'water', 'watch', 'amazon', 'spell', 'kiwi', 'ghost',
            'cold', 'doctor', 'port', 'bark', 'foot', 'luck', 'nail', 'ice', 'needle', 'disease', 'comic', 'pool',
            'field', 'star', 'cycle', 'shadow', 'fan', 'compound', 'heart', 'flute', 'millionaire', 'pyramid', 'africa',
            'robin', 'chest', 'casino', 'fish', 'oil', 'alps', 'brush', 'march', 'mint', 'dance', 'snowman', 'torch',
            'round', 'wake', 'satellite', 'calf', 'head', 'ground', 'club', 'ruler', 'tie', 'parachute', 'board',
            'paste', 'lock', 'knight', 'pit', 'fork', 'egypt', 'whale', 'scale', 'knife', 'plate', 'scorpion', 'bottle',
            'boom', 'bolt', 'fall', 'draft', 'hotel', 'game', 'mount', 'train', 'air', 'turkey', 'root', 'charge',
            'space', 'cat', 'olive', 'mouse', 'ham', 'washer', 'pound', 'fly', 'server', 'shop', 'engine', 'himalayas',
            'box', 'antarctica', 'shoe', 'tap', 'cross', 'rose', 'belt', 'thumb', 'gold', 'point', 'opera', 'pirate',
            'tag', 'olympus', 'cotton', 'glove', 'sink', 'carrot', 'jack', 'suit', 'glass', 'spot', 'straw', 'well',
            'pan', 'octopus', 'smuggler', 'grass', 'dwarf', 'hood', 'duck', 'jet', 'mercury',
        ]

    _a, word_indexing_dict, _b = create_embedder(method)
    return list(word_indexing_dict.keys())


def intersect_vocabularies(*vocabs):
    voc_generator = map(get_vocabulary, vocabs)
    return list(set(next(voc_generator)).intersection(*voc_generator))


def get_embedder_with_vocab(embed_method: str = 'GloVe', vocab_method: str = 'wordnet-nouns'):
    filename = f"embed_{embed_method}_vocab_{vocab_method}.pkl"
    x = load_saved_embedding(filename)
    if x is not None:
        embeddings, word_indexing_dict, index_word_dict, vocab = x
    else:
        embeddings, word_indexing_dict, _ = create_embedder(embed_method)
        all_vocab = get_vocabulary(vocab_method)
        vocab = [v for v in all_vocab if v in word_indexing_dict]
        print(f"{100 * len(vocab) / len(all_vocab)}% of the vocabulary \'{vocab_method}\' can be used with"
              f" \'{embed_method}\'embedding. Total of {len(vocab)} words")
        vocab_indices = [word_indexing_dict[word] for word in vocab]
        embeddings = embeddings[vocab_indices + [-1]]
        word_indexing_dict = {v: i for i, v in enumerate(vocab)}
        index_word_dict = {i: v for i, v in enumerate(vocab)}

        dump_embeddings(filename, (embeddings, word_indexing_dict, index_word_dict, vocab))
    return embeddings, word_indexing_dict, index_word_dict, vocab


def triple_restricted_vocab(embed_method_1: str, embed_method_2: str, vocab_method: str):
    """ restrict the vocabulary to all three methods """
    t0 = time.perf_counter()
    *_, v1 = get_embedder_with_vocab(embed_method_1, vocab_method)
    if embed_method_1 == embed_method_2:
        return v1
    *_, v2 = get_embedder_with_vocab(embed_method_2, vocab_method)
    result = list(set(v1).intersection(set(v2)))
    print("time for triple_restriced_vocab:", time.perf_counter() - t0)
    return result


class EmbeddingAgent(nn.Module):
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'GloVe',
                 vocab_method: str = 'wordnet-nouns', dist_metric: str = 'l2'):
        super(EmbeddingAgent, self).__init__()
        self.gca = good_cards_amount
        self.tca = total_cards_amount
        self.embeddings, self.word_index_dict, self.index_word_dict, _ = get_embedder_with_vocab(embedding_method,
                                                                                                 vocab_method)
        assert dist_metric in ['cosine_sim', 'l2']
        if dist_metric == 'cosine_sim':
            # Normalize the embedding vectors. Euclidean norm is now rank-equivalent to cosine similarity, so no
            # further change is needed.
            embedding_norms = torch.norm(self.embeddings, dim=-1, keepdim=True)
            self.embeddings = self.embeddings / embedding_norms

    def embedder(self, word: str):
        return self.embeddings[self.word_index_dict.get(word, -1)]

    def known_word(self, word: str):
        return word in self.word_index_dict


def try_ce_loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
    loss = F.binary_cross_entropy(receiver_output, sender_input.view(-1, 784), reduction='none').mean(dim=1)
    return loss, {}

#########################

# def main():
#     gca = 4
#     tca = 10
#     embedding_method = 'GloVe'
#     vocab_method = 'wordnet-nouns'
#     rec_input_size = 400
#     kwargs = dict(total_cards=tca, good_cards=gca, embedding_method=embedding_method, vocab_method=vocab_method)
#     opts = core.init(params=['--random_seed=7',  # will initialize numpy, torch, and python RNGs
#                              '--lr=1e-3',  # sets the learning rate for the selected optimizer
#                              '--batch_size=32',
#                              '--optimizer=adam'])
#
#     train_loader = CodenameDataLoader(tca, gca, embedding_method, vocab_method)
#     test_loader = CodenameDataLoader(tca, gca, embedding_method, vocab_method)
#     sender = TrainableSender(**kwargs)
#     vocab_size = sender.vocab_size
#     sender = core.GumbelSoftmaxWrapper(sender, temperature=1.0)
#     receiver = TrainableReceiver(**kwargs, input_size=rec_input_size)
#     receiver = core.SymbolReceiverWrapper(receiver, vocab_size, agent_input_size=rec_input_size)
#
#     game = core.SymbolGameGS(sender, receiver, rotem_utils.try_ce_loss)
#     optimizer = torch.optim.Adam(game.parameters())
#     trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
#                            callbacks=[core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)])
#     trainer.train(n_epochs=1)


# class TrainableSender(rotem_utils.EmbeddingAgent):
#     def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str):
#         super(TrainableSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method)
#         self.vocab_size = len(self.word_index_dict)
#         self.fc = nn.Linear(100, self.vocab_size)
#
#     def forward(self, x, aux_input=None):
#         print(f"sender inputs: \n \t x: {x} \n \t aux: {aux_input} \n \n")
#         return None
#
#
# class TrainableReceiver(rotem_utils.EmbeddingAgent):
#     def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, input_size: int):
#         super(TrainableReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method)
#         self.fc = nn.Linear(input_size, 10)     # 10 tbd
#
#     def forward(self, channel_input, receiver_input, aux_input=None):
#         print(f"Receiver inputs: \n \t channel_input: {channel_input} \n \t receiver_input: {receiver_input}"
#               f" \n \t aux: {aux_input} \n \n")
#         return None
