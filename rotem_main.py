import torch
import torch.nn as nn
from torch.nn import functional as F

import egg.core as core
from transformers import BertTokenizer, BertModel
import rotem_utils

import random
from itertools import combinations
from typing import List
from scipy.cluster.vq import kmeans2
import time


class EmbeddingClusterSender(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str):
        super(EmbeddingClusterSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method)
        self.cluster_amounts = [self.tca // (i + 1) for i in range(4)]
        self.good_embeds = None
        self.bad_embeds = None

    def find_closest_word(self, centroid, words, verbose: bool = False) -> str:
        t0 = time.perf_counter()
        norm_dif = torch.norm(self.embeddings[:-1] - centroid, dim=1)
        norm_dif[[self.word_index_dict[word] for word in words]] = torch.inf
        index = torch.argmin(norm_dif).item()
        closest_word = self.index_word_dict[index]
        if verbose:
            print(f"Time to find closest word to centroid: {time.perf_counter() - t0}")
        return closest_word

    def largest_good_cluster(self):
        largest_cluster_size = 0
        largest_cluster_indices = None
        largest_cluster_centroid = None
        for cluster_amount in self.cluster_amounts:
            data = torch.cat([self.good_embeds, self.bad_embeds], dim=0)
            centroids, labels = kmeans2(data, cluster_amount)
            labels = torch.from_numpy(labels)
            good_cluster_labels = set(labels[:self.gca].tolist()) - set(labels[self.gca:].tolist())
            for label in good_cluster_labels:
                indices = (labels == label).nonzero(as_tuple=True)[0]
                if len(indices) > largest_cluster_size:
                    largest_cluster_size = len(indices)
                    largest_cluster_indices = indices
                    largest_cluster_centroid = centroids[label]
        if largest_cluster_size == 0:
            print("receiver didn't find any cluster!")
        return largest_cluster_centroid, largest_cluster_indices, largest_cluster_size

    def forward(self, words: tuple, verbose: bool = False):
        """
        words = (good_words, bad_words), which are lists of strings.
        """
        good_words, bad_words = words
        g_embeds_list = list(map(self.embedder, good_words))
        self.good_embeds = torch.stack(g_embeds_list)
        b_embeds_list = list(map(self.embedder, bad_words))
        self.bad_embeds = torch.stack(b_embeds_list)

        centroid, indices, clue_len = self.largest_good_cluster()
        clue_word = self.find_closest_word(centroid, good_words + bad_words, verbose=verbose)
        if verbose:
            chosen_words = [good_words[i] for i in indices]
            print(f"for {clue_len} chosen words: \n \t {chosen_words} \n the clue is {clue_word}")

        return clue_word, clue_len


class ExhaustiveSearchSender(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str):
        super(ExhaustiveSearchSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method)

    def forward(self, words: tuple, verbose: bool = False):
        """
        words = (good_words, bad_words), which are lists of strings.
        """
        good_words, bad_words = words
        g_embeds_list = list(map(self.embedder, good_words))
        good_embeds = torch.stack(g_embeds_list)
        b_embeds_list = list(map(self.embedder, bad_words))
        bad_embeds = torch.stack(b_embeds_list)

        good_diff = torch.norm(self.embeddings.unsqueeze(1) - good_embeds, dim=-1)  # (N_vocab, num_good_cards)
        bad_diff = torch.norm(self.embeddings.unsqueeze(1) - bad_embeds, dim=-1)  # (N_vocab, num_bad_cards)

        closest_bad_distance = bad_diff.min(dim=1, keepdim=True)[0]  # distance to the closest bad card - shape (N, 1)
        boolean_closeness_mat = (good_diff < closest_bad_distance)
        close_good_cards = boolean_closeness_mat.sum(dim=1)  # amount of good cards that are closer than the closest
        # bad card - shape (N_vocab)
        close_good_cards[[self.word_index_dict[word] for word in good_words + bad_words]] = 0   # to avoid

        best_word_cluesize, best_word_idx = torch.max(close_good_cards, 0)
        clue_word = self.index_word_dict[best_word_idx.item()]
        if verbose:
            cluster_words_indices = boolean_closeness_mat[best_word_idx].nonzero(as_tuple=True)[0]
            # print(f"For the {best_word_cluesize.item()} "
            #       f"good word: \n \t {[good_words[i] for i in cluster_words_indices]}\n \t The chosen clue is {clue_word}")
            print(f"target words: {sorted([good_words[i] for i in cluster_words_indices])} \n \t guess: {clue_word}")
        return clue_word, best_word_cluesize.item()


class EmbeddingNearestReceiver(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str):
        super(EmbeddingNearestReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method)

    def forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        clue_word, clue_amount = clue
        clue_array = self.embedder(clue_word)
        words_array = torch.stack(list(map(self.embedder, words)))
        norm_dif = torch.norm(words_array - clue_array, dim=1)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return {words[i] for i in indices}


class CodenameDataLoader:
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embed_method: str = 'GloVe',
                 vocab_method: str = 'wordnet-nouns'):
        """
        Note: the embed_method is only used to filter the vocabulary in this class.
        """
        *_, self.vocab = rotem_utils.get_embedder_with_vocab(embed_method, vocab_method)
        self.vocab_length = len(self.vocab)
        self.random_mask = torch.zeros(self.vocab_length)
        self.random_mask[:1000] = 1
        self.gca = good_cards_amount
        self.tca = total_cards_amount

    def __iter__(self):
        while True:
            indices = torch.multinomial(torch.ones(self.vocab_length), self.tca)
            words = [self.vocab[i] for i in indices]
            # words = ['inspector', 'suggestion', 'police', 'emphasis', 'member', 'meal', 'two', 'independence', 'addition',
            #          'person', 'hall', 'love', 'shopping', 'presentation', 'director']

            g_words, b_words = words[:self.gca], words[self.gca:]
            sender_input = (g_words, b_words)
            receiver_input = random.sample(words, len(words))   # shuffled words
            yield sender_input, receiver_input


class TrainableSender(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str):
        super(TrainableSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method)
        self.vocab_size = len(self.word_index_dict)
        self.fc = nn.Linear(100, self.vocab_size)

    def forward(self, x, aux_input=None):
        print(f"sender inputs: \n \t x: {x} \n \t aux: {aux_input} \n \n")
        return None


class TrainableReceiver(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, input_size: int):
        super(TrainableReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method)
        self.fc = nn.Linear(input_size, 10)     # 10 tbd

    def forward(self, channel_input, receiver_input, aux_input=None):
        print(f"Receiver inputs: \n \t channel_input: {channel_input} \n \t receiver_input: {receiver_input}"
              f" \n \t aux: {aux_input} \n \n")
        return None


def skyline_main():
    gca = 10
    tca = 20
    embedding_method = 'GloVe'
    vocab_method = 'wordnet-nouns'
    kwargs = dict(total_cards=tca, good_cards=gca, embedding_method=embedding_method, vocab_method=vocab_method)
    trial_amount = 6

    sender = ExhaustiveSearchSender(**kwargs)
    # sender = EmbeddingClusterSender(**kwargs)
    receiver = EmbeddingNearestReceiver(**kwargs)
    dataloader = CodenameDataLoader(tca, gca, embedding_method, vocab_method)
    i=0
    for sender_input, receiver_input in dataloader:
        # print(f"receiver_input - all words: {receiver_input}\n")
        # print(f"sender input:\n \tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
        print(f"\tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
        t0 = time.perf_counter()
        clue = sender(sender_input, verbose=True)
        print("sender time:", time.perf_counter() - t0)
        choice = receiver(receiver_input, clue)
        print("guess:", sorted(choice), "\n\n")
        i += 1
        if i == trial_amount:
            break


def main():
    gca = 4
    tca = 10
    embedding_method = 'GloVe'
    vocab_method = 'wordnet-nouns'
    rec_input_size = 400
    kwargs = dict(total_cards=tca, good_cards=gca, embedding_method=embedding_method, vocab_method=vocab_method)
    opts = core.init(params=['--random_seed=7',  # will initialize numpy, torch, and python RNGs
                             '--lr=1e-3',  # sets the learning rate for the selected optimizer
                             '--batch_size=32',
                             '--optimizer=adam'])

    train_loader = CodenameDataLoader(tca, gca, embedding_method, vocab_method)
    test_loader = CodenameDataLoader(tca, gca, embedding_method, vocab_method)
    sender = TrainableSender(**kwargs)
    vocab_size = sender.vocab_size
    sender = core.GumbelSoftmaxWrapper(sender, temperature=1.0)
    receiver = TrainableReceiver(**kwargs, input_size=rec_input_size)
    receiver = core.SymbolReceiverWrapper(receiver, vocab_size, agent_input_size=rec_input_size)

    game = core.SymbolGameGS(sender, receiver, rotem_utils.try_ce_loss)
    optimizer = torch.optim.Adam(game.parameters())
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
                           callbacks=[core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)])
    trainer.train(n_epochs=1)

if __name__ == '__main__':
    skyline_main()
