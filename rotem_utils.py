import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import os
from os.path import join, isdir, isfile
from tqdm import tqdm
import json
import requests

import nltk
from nltk.corpus import words, wordnet as wn
import gensim.downloader as api

from agents import EmbeddingAgent, ContextEmbeddingAgent

# global variables - names of directories and files
embedding_saves_dir = "embeddings_saved"
vocabs_saves_dir = "vocabs_saved"
glove_embeddings_file = "saved_glove_embeddings.pkl"
word2vec_embeddings_file = "saved_w2v_embeddings.pkl"
ROTEM_RESULTS_DIR = 'rotem_results'


def load_saved_data(file_name: str, directory: str):
    assert directory in [embedding_saves_dir, vocabs_saves_dir]
    if not isdir(directory):  # no directory 'embeddings_saved'
        os.mkdir(directory)
        return None

    if not isfile(path := join(directory, file_name)):  # no file file_name in the directory
        return None

    with open(path, "rb") as read_file:
        x = pickle.load(read_file)
    return x


def dump_data(file_name, obj, directory: str):
    assert directory in [embedding_saves_dir, vocabs_saves_dir]
    assert isdir(directory)
    with open(join(directory, file_name), "wb") as write_file:
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
        dump_data(glove_embeddings_file, glove_data, embedding_saves_dir)
    return glove_data


def initiate_word2vec(save_data=True):
    gensim_model = api.load('word2vec-google-news-300')
    word_indexing_dict = gensim_model.key_to_index
    embeddings = torch.from_numpy(gensim_model.vectors)
    unk_vector = embeddings.mean(dim=0, keepdim=True)
    embeddings = torch.cat([embeddings, unk_vector], dim=0)
    w2v_data = (embeddings, word_indexing_dict)
    if save_data:
        dump_data(word2vec_embeddings_file, w2v_data, embedding_saves_dir)
    return w2v_data


def create_embedder(method: str):
    if method == 'GloVe':
        x = load_saved_data(glove_embeddings_file, embedding_saves_dir)
        if x is None:
            embeddings, word_indexing_dict = initiate_GloVe()
        else:
            embeddings, word_indexing_dict = x
    elif method == 'word2vec':
        x = load_saved_data(word2vec_embeddings_file, embedding_saves_dir)
        if x is None:
            embeddings, word_indexing_dict = initiate_word2vec()
        else:
            embeddings, word_indexing_dict = x
    else:
        raise ValueError("currently, only GloVe and word2vec embeddings are allowed, not " + method)

    index_word_dict = {v: k for k, v in word_indexing_dict.items()}

    return embeddings, word_indexing_dict, index_word_dict


def get_vocabulary(method: str):
    filename = f"{method}_vocab.pkl"
    v = load_saved_data(filename, vocabs_saves_dir)
    if v is not None:
        return v

    if method == 'nltk-words':
        try:
            x = words.words()
        except LookupError:
            nltk.download('words')
            x = words.words()

    elif method == 'wordnet-nouns':
        try:
            x = wn.all_synsets('n')
        except LookupError:
            nltk.download('wordnet')
            try:
                x = wn.all_synsets('n')
            except LookupError:
                nltk.download('omw-1.4')
                x = wn.all_synsets('n')
        x = list(map(lambda s: s.name().partition('.')[0], x))

    elif method == 'nouns-kaggle':
        # taken from https://www.kaggle.com/datasets/leite0407/list-of-nouns
        x = ['atm'] + np.char.lower(pd.read_csv('nounlist.csv').to_numpy().reshape(-1).astype(str)).tolist()

    elif method == 'codenames-words':
        x = [
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
    else:
        _a, word_indexing_dict, _b = create_embedder(method)
        x = list(word_indexing_dict.keys())

    dump_data(file_name=filename, obj=x, directory=vocabs_saves_dir)
    return x


def intersect_vocabularies(*vocabs):
    vocabs = [x for x in vocabs if x not in ['bert', 'deberta']]
    voc_generator = map(get_vocabulary, set(vocabs))
    return list(set(next(voc_generator)).intersection(*voc_generator))


def get_embedder_with_vocab(embed_method: str = 'GloVe', vocab_method: str = 'wordnet-nouns'):
    filename = f"embed_{embed_method}_vocab_{vocab_method}.pkl"
    x = load_saved_data(filename, embedding_saves_dir)
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

        dump_data(filename, (embeddings, word_indexing_dict, index_word_dict, vocab), embedding_saves_dir)
    return embeddings, word_indexing_dict, index_word_dict, vocab


def get_embedder_list_vocab(embed_method: str, vocab: list):
    embeddings, word_indexing_dict, index_word_dict = create_embedder(embed_method)
    vocab = [word for word in vocab if word in word_indexing_dict]
    indices = [word_indexing_dict[word] for word in vocab]
    embeddings = embeddings[indices]
    word_indexing_dict = {word: index for index, word in enumerate(vocab)}
    index_word_dict = {index: word for index, word in enumerate(vocab)}
    return embeddings, word_indexing_dict, index_word_dict


# def triple_restricted_vocab(embed_method_1: str, embed_method_2: str, vocab_method: str):
#     """ restrict the vocabulary to all three methods """
#     t0 = time.perf_counter()
#     *_, v1 = get_embedder_with_vocab(embed_method_1, vocab_method)
#     if embed_method_1 == embed_method_2:
#         return v1
#     *_, v2 = get_embedder_with_vocab(embed_method_2, vocab_method)
#     result = list(set(v1).intersection(set(v2)))
#     print("time for triple_restriced_vocab:", time.perf_counter() - t0)
#     return result


def BiSANLM_score(model, tokenizer, sentence: str):
    """ http://proceedings.mlr.press/v101/shin19a/shin19a.pdf """
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    tokenized_ids = tokenizer(sentence, return_tensors='pt')['input_ids'].view(-1)
    labels = tokenized_ids[1:-1]
    tokens_amount = len(tokenized_ids) - 2
    arange = torch.arange(tokens_amount)
    input_matrix = torch.vstack([tokenized_ids]*tokens_amount)  # (tokens_amount, tokens_amount + 2)
    input_matrix[arange, arange + 1] = mask_id
    with torch.inference_mode():
        outputs = model(input_matrix)['logits']     # (tokens_amount, tokens_amount + 2, V)
    # take the masked entries
    outputs = outputs[arange, arange + 1, :]  # (tokens_amount, V)
    # log likelihood
    outputs = F.log_softmax(outputs, dim=1)
    scores = outputs[arange, labels]
    return scores.sum().item()


def ce_score(model, tokenizer, sentence: str):
    tokenized_ids = tokenizer(sentence, return_tensors='pt')['input_ids']
    with torch.inference_mode():
        outputs = model(tokenized_ids)['logits']
    score = F.cross_entropy(outputs.squeeze(), tokenized_ids.squeeze())
    return -score


def bert_sentence_scorer(model, tokenizer, w1: str, w2: str, method: str = 'ce'):
    assert method in ['ce', 'BiS'], f"illegal method {method}"
    score_func = {'ce': ce_score, 'BiS': BiSANLM_score}[method]
    templates = [
        f"{w1} is a type of {w2}",
        f"{w2} is a type of {w1}",
        f"you can find {w1} in {w2}",
        f"you can find {w2} in {w1}",
        f"{w1} and {w2} are types of [MASK]",
        f"{w1} {w2}",
        f"{w2} {w1}",
        f"{w1} is a synonym of {w2}"
    ]
    best_score = None
    arg_best = None
    for sent in templates:
        score = score_func(model, tokenizer, sent)
        if best_score is None or score > best_score:
            best_score = score
            arg_best = sent
    return arg_best, best_score


def few_shot_huggingface(prompt: str, delimiter: str = "###"):
    API_TOKEN = "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"

    parameters = {
        'max_new_tokens': 25,  # number of generated tokens
        'temperature': 0.0001,  # controlling the randomness of generations
        'end_sequence': delimiter  # stopping sequence for generation
    }

    options = {'use_cache': False}

    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    body = {"inputs": prompt, 'parameters': parameters, 'options': options}
    response = requests.request("POST", API_URL, headers=headers, data=json.dumps(body))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        return "Error:" + " ".join(response.json()['error'])
    else:
        return response.json()[0]['generated_text'][len(prompt):]


def average_rank_embedding_distance(agent1: EmbeddingAgent, agent2: EmbeddingAgent, vocab: list = None):
    if vocab is None:
        assert agent1.vocab == agent2.vocab, "can only compare embeddings with the same vocabulary unless given one"
        vocab = agent1.vocab
    N = len(vocab)
    embedding1 = agent1.embedder(vocab)
    embedding2 = agent2.embedder(vocab)
    diff = 0
    for i in tqdm(range(N), total=N):
        word_array = embedding1[i]
        norm_diff = torch.norm(embedding1 - word_array, dim=1)
        sorted_indices_1 = norm_diff.argsort()

        word_array = embedding2[i]
        norm_diff = torch.norm(embedding2 - word_array, dim=1)
        sorted_indices_2 = norm_diff.argsort()
        new_diff = (sorted_indices_1 - sorted_indices_2).abs().sum() / len(vocab)
        diff += new_diff.item()
    diff /= N
    return diff


def noise_experiment_plots():
    df = pd.read_csv(os.path.join(ROTEM_RESULTS_DIR, 'noise_experiment_final'))
    noises = df['noise']
    for y in ['norm_difference', 'metric_difference', 'game_difference']:
        y_name = y.replace('_', ' ')
        plt.plot(noises, df[y])
        plt.title(f"{y_name} as a function of noise")
        plt.xlabel("noise")
        plt.ylabel(y_name)
        plt.show()

    for y in ['norm_difference', 'metric_difference', 'game_difference']:
        y_name = y.replace('_', ' ')
        plt.semilogx(noises, df[y])
        plt.title(f"{y_name} as a function of noise, log-x")
        plt.xlabel("noise")
        plt.ylabel(y_name)
        plt.show()


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
