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
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(EmbeddingClusterSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method, dist_metric)
        cluster_range = 4
        multiplicity = 4
        self.cluster_amounts = sum([[self.tca // (i + 1)]*multiplicity for i in range(cluster_range)], start=[])
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
            targets = sorted([good_words[i] for i in indices])
        else:
            targets = None

        return clue_word, clue_len, targets


class ExhaustiveSearchSender(rotem_utils.EmbeddingAgent):
    """
    A sender agent which operates as follows - given good and bad cards, finds the set of all words in the vocabulary
    with maximal amount of good cards closer than all bad cards. Out of this set, the agent chosses the word which is
    closest to it's matching good cards.
    """
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(ExhaustiveSearchSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method, dist_metric)

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
        close_good_cards[[self.word_index_dict[word] for word in good_words + bad_words]] = 0   # avoid current cards

        # new - choose the best option among the optimal clues ####################
        all_best_indices = torch.nonzero(close_good_cards == close_good_cards.max()).view(-1)
        best_word_cluesize = close_good_cards[all_best_indices[0]]
        print("amount of optimal clues:", len(all_best_indices))
        best_good_diff = good_diff[all_best_indices]
        closest_bad_distance = closest_bad_distance[all_best_indices]
        best_good_diff = torch.where(best_good_diff < closest_bad_distance,
                                     best_good_diff,
                                     torch.zeros(best_good_diff.shape, dtype=torch.float)).sum(dim=1)
        best_word_idx = torch.argmin(best_good_diff)        # index of best word among optimals
        best_word_idx = all_best_indices[best_word_idx]     # that word's index in the entire vocabulary
        ######################################################

        # best_word_cluesize, best_word_idx = torch.max(close_good_cards, 0)
        clue_word = self.index_word_dict[best_word_idx.item()]
        if verbose:
            cluster_words_indices = boolean_closeness_mat[best_word_idx].nonzero(as_tuple=True)[0]
            targets = sorted([good_words[i] for i in cluster_words_indices])
        else:
            targets = None
        return clue_word, best_word_cluesize.item(), targets


class EmbeddingNearestReceiver(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(EmbeddingNearestReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method, dist_metric)

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
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embed_method1: str = 'GloVe',
                 embed_method2: str = 'GloVe', vocab_method: str = 'wordnet-nouns'):
        """
        Note: the embed_method is only used to filter the vocabulary in this class.
        """
        self.vocab = rotem_utils.triple_restricted_vocab(embed_method1, embed_method2, vocab_method)
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

    def legal_word(self, word):
        return word in self.vocab


def skyline_main(tca, gca, sender_type, sender_emb_method, receiver_emb_method, vocab_method, dist_metric,
                 trial_amount, verbose):
    """

    Parameters
    ----------
    tca - total cards amount
    gca - good cards amount
    sender_type - 'exhaustive' or 'cluster', which sender class to use
    sender_emb_method - 'GloVe' or 'word2vec'
    receiver_emb_method - 'GloVe' or 'word2vec'
    vocab_method - 'GloVe' or 'word2vec' or 'nltk-words' or 'wordnet-nouns' or 'nouns-kaggle'
    dist_metric - 'l2' or 'cosine_sim'
    trial_amount - how many games to run for the evaluation
    verbose - bool

    Returns
    -------

    """
    kwargs = dict(total_cards=tca, good_cards=gca, vocab_method=vocab_method, dist_metric=dist_metric)

    sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[sender_type]
    sender = sender_class(**kwargs, embedding_method=sender_emb_method)
    receiver = EmbeddingNearestReceiver(**kwargs, embedding_method=receiver_emb_method)
    dataloader = CodenameDataLoader(tca, gca, sender_emb_method, receiver_emb_method, vocab_method)
    i=0
    for sender_input, receiver_input in dataloader:
        # print(f"receiver_input - all words: {receiver_input}\n")
        # print(f"sender input:\n \tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
        # print(f"\tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
        # t0 = time.perf_counter()
        *clue, targets = sender(sender_input, verbose=verbose)
        # print(f"target words: {targets} \n \t clue: {clue[0]} \n")
        # print("sender time:", time.perf_counter() - t0)
        choice = receiver(receiver_input, clue)
        print(f"clue size {clue[1]}, guess success: {100 * len([x for x in choice if x in targets]) / len(targets)}%")
        # print("guess:", sorted(choice), "\n\n")
        i += 1
        if i == trial_amount:
            break


def interactive_game():
    print("This is an interactive codenames game. You have two options: \n"
          "1) Guess words using a clue generated by our sender \n"
          "2) Choose a clue and give it to our receiver \n \n")
    game_choice = rotem_utils.user_choice('1', "Which option do you choose?", options=['1', '2'], full_text=True)
    print("now you choose the game parameters. To take the default, enter \'a\' at any decision. \n")

    tca = rotem_utils.user_choice(10, "total cards amount", demand_int=True)
    gca = rotem_utils.user_choice(4, "good cards amount", demand_int=True)
    vocab_method = rotem_utils.user_choice('nouns-kaggle', "vocabulary limitation",
                                           options=['GloVe', 'word2vec', 'nltk-words', 'wordnet-nouns', 'nouns-kaggle'])
    dist_metric = rotem_utils.user_choice('cosine_sim', "distance metric", options=['l2', 'cosine_sim'])
    kwargs = dict(total_cards=tca, good_cards=gca, vocab_method=vocab_method, dist_metric=dist_metric)

    if game_choice == '1':
        sender_type = rotem_utils.user_choice('exhaustive', "sender type", options=['exhaustive', 'cluster'])
        emb_method = rotem_utils.user_choice('GloVe', "sender embedding method", options=['GloVe', 'word2vec'])
        dataloader = CodenameDataLoader(tca, gca, emb_method, emb_method, vocab_method)
        data_generator = iter(dataloader)
        sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[sender_type]
        sender = sender_class(**kwargs, embedding_method=emb_method)

        while True:
            sender_input, receiver_input = next(data_generator)
            print(f"words on the board: \n {receiver_input} \n")
            clue, clue_num, targets = sender(sender_input, verbose=True)
            print(f"The sender gives the clue \"{clue}\", for {clue_num} words. Can you guess them?")
            guesses = []
            for i in range(clue_num):
                guess = rotem_utils.user_choice('stop', f"guess number {i+1}:", options=['stop'] + receiver_input,
                                                full_text=True)
                if guess == 'stop':
                    break
                guesses.append(guess)
            guesses.sort()
            print(f"your guesses: \t {guesses} \n you got right: \t {[w for w in guesses if w in targets]} \n"
                  f"you got wrong: \t {[w for w in guesses if w not in targets]} \n the true words: \t {targets}")
            answer = rotem_utils.user_choice('yes', "play again?", options=['yes', 'no'], full_text=True)
            if answer == 'no':
                break
    else:   # game_choice == '2'
        emb_method = rotem_utils.user_choice('GloVe', "receiver embedding method", options=['GloVe', 'word2vec'])
        dataloader = CodenameDataLoader(tca, gca, emb_method, emb_method, vocab_method)
        receiver = EmbeddingNearestReceiver(**kwargs, embedding_method=emb_method)
        data_generator = iter(dataloader)
        while True:
            sender_input, receiver_input = next(data_generator)
            print(f"\tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
            print("What is your clue? \n")
            while True:
                clue = input()
                if dataloader.legal_word(clue):
                    break
                print("illegal word. try again. \n")
            clue_num = rotem_utils.user_choice(1, "number of guesses", demand_int=True)
            assert clue_num <= tca, "too many guesses required"
            choice = receiver(receiver_input, (clue, clue_num))
            print(f"Our receiver guessed: \n \t {sorted(choice)} - "
                  f"{100 * len([x for x in choice if x in sender_input[0]]) / clue_num}% correct.")
            answer = rotem_utils.user_choice('yes', "play again?", options=['yes', 'no'], full_text=True)
            if answer == 'no':
                break


if __name__ == '__main__':
    # skyline_main(tca=10, gca=5,
    #              sender_type='exhaustive', sender_emb_method='GloVe', receiver_emb_method='word2vec',
    #              vocab_method='nouns-kaggle', dist_metric='cosine_sim',
    #              trial_amount=10, verbose=True)
    interactive_game()
