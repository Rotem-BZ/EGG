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


class EmbeddingClusterSender(nn.Module):
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'GloVe', **kwargs):
        super(EmbeddingClusterSender, self).__init__()
        self.gca = good_cards_amount
        self.tca = total_cards_amount
        self.embeddings, self.word_index_dict, self.index_word_dict = rotem_utils.create_embedder(embedding_method)
        assert self.tca > 4
        self.cluster_amounts = [self.tca - 1, self.tca - 2, self.tca - 3]  # TODO: find better function
        self.good_embeds = None
        self.bad_embeds = None

    def embedder(self, word: str):
        return self.embeddings[self.word_index_dict.get(word, -1)]

    def find_closest_word(self, centroid, verbose: bool = False) -> str:
        t0 = time.perf_counter()
        norm_dif = torch.norm(self.embeddings[:-1] - centroid, dim=1)
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
        clue_word = self.find_closest_word(centroid, verbose=verbose)
        if verbose:
            chosen_words = [good_words[i] for i in indices]
            print(f"for {clue_len} chosen words: \n \t {chosen_words} \n the clue is {clue_word}")

        return clue_word, clue_len


class ExhaustiveSearchSender(nn.Module):
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'GloVe', **kwargs):
        super(ExhaustiveSearchSender, self).__init__()
        self.embeddings, self.word_index_dict, self.index_word_dict = rotem_utils.create_embedder(embedding_method)

    def embedder(self, word: str):
        return self.embeddings[self.word_index_dict.get(word, -1)]

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

        best_word_cluesize, best_word_idx = torch.max(close_good_cards, 0)
        clue_word = self.index_word_dict[best_word_idx.item()]
        if verbose:
            cluster_words_indices = boolean_closeness_mat[best_word_idx].nonzero(as_tuple=True)[0]
            print(f"For the {best_word_cluesize.item()} "
                  f"good word: \n \t {[good_words[i] for i in cluster_words_indices]}\n \t The chosen clue is {clue_word}")
        return clue_word, best_word_cluesize.item()


class EmbeddingNearestReceiver(nn.Module):
    def __init__(self, embedding_method: str = 'GloVe'):
        super(EmbeddingNearestReceiver, self).__init__()
        self.embeddings, self.word_index_dict, self.index_word_dict = rotem_utils.create_embedder(embedding_method)

    def embedder(self, word: str):
        return self.embeddings[self.word_index_dict.get(word, -1)]

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
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'GloVe', **kwargs):
        self.vocab = rotem_utils.get_vocabulary(embedding_method)
        self.vocab_length = len(self.vocab)
        self.random_mask = torch.zeros(self.vocab_length)
        self.random_mask[:10000] = 1
        self.gca = good_cards_amount
        self.tca = total_cards_amount

    def __iter__(self):
        indices = torch.multinomial(torch.ones(self.vocab_length), self.tca)
        words = [self.vocab[i] for i in indices]
        g_words, b_words = words[:self.gca], words[self.gca:]
        sender_input = (g_words, b_words)
        receiver_input = random.sample(words, len(words))
        yield sender_input, receiver_input


def main():
    gca = 4
    tca = 15
    embedding_method = 'GloVe'

    sender = ExhaustiveSearchSender(tca, gca, embedding_method)
    receiver = EmbeddingNearestReceiver(embedding_method)
    dataloader = CodenameDataLoader(tca, gca, embedding_method)
    for sender_input, receiver_input in dataloader:
        print(f"receiver_input - all words: {receiver_input}\n")
        print(f"sender input:\n \tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
        t0 = time.perf_counter()
        clue = sender(sender_input, verbose=True)
        print("sender time:", time.perf_counter() - t0)
        choice = receiver(receiver_input, clue)
        print("receiver choice:", choice)
        break


if __name__ == '__main__':
    main()
