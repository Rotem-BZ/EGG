import pandas as pd
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
from tqdm import tqdm


class EmbeddingClusterSender(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(EmbeddingClusterSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method,
                                                     dist_metric)
        cluster_range = 4
        multiplicity = 4
        self.cluster_amounts = sum([[self.tca // (i + 1)] * multiplicity for i in range(cluster_range)], start=[])
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
        self.good_embeds = self.embedder(good_words)
        self.bad_embeds = self.embedder(bad_words)

        centroid, indices, clue_len = self.largest_good_cluster()
        clue_word = self.find_closest_word(centroid, good_words + bad_words, verbose=verbose)
        targets = sorted([good_words[i] for i in indices])

        return clue_word, clue_len, targets, {}


class ExhaustiveSearchSender(rotem_utils.EmbeddingAgent):
    """
    A sender agent which operates as follows - given good and bad cards, finds the set of all words in the vocabulary
    with maximal amount of good cards closer than all bad cards. Out of this set, the agent chosses the word which is
    closest to it's matching good cards.
    """

    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str,
                 tie_break_method: str = 'avg_blue_dist'):
        super(ExhaustiveSearchSender, self).__init__(total_cards, good_cards, embedding_method, vocab_method,
                                                     dist_metric)
        assert tie_break_method in ['avg_blue_dist', 'max_blue_dist', 'max_radius', 'red_blue_diff']
        """
        avg_blue_dist - minimize the average distance/dissimilarity to the targets
        max_blue_dist - minimize the maximal distance/dissimilarity to the targets
        max_radius - maximize the distance to the closest red card
        red_blue_diff - maximize the margin between targets and red cards
        """
        self.tie_break = tie_break_method

    def tie_break_scores(self, target_diffs, closest_bad_dist):
        if self.tie_break == 'avg_blue_dist':
            return target_diffs.sum(dim=1)
        if self.tie_break == 'max_blue_dist':
            return target_diffs.max(dim=1)[0]
        if self.tie_break == 'max_radius':
            return -closest_bad_dist
        if self.tie_break == 'red_blue_diff':
            return target_diffs.max(dim=1)[0] - closest_bad_dist
        raise ValueError("illegal tie break " + self.tie_break)

    def forward(self, words: tuple, verbose: bool = False):
        """
        words = (good_words, bad_words), which are lists of strings.
        """
        good_words, bad_words = words
        good_embeds = self.embedder(good_words)
        bad_embeds = self.embedder(bad_words)

        good_diff = torch.norm(self.embeddings[:-1].unsqueeze(1) - good_embeds, dim=-1)  # (N_vocab, num_good_cards)
        bad_diff = torch.norm(self.embeddings[:-1].unsqueeze(1) - bad_embeds, dim=-1)  # (N_vocab, num_bad_cards)

        closest_bad_distance = bad_diff.min(dim=1, keepdim=True)[0]  # distance to the closest bad card - shape (N, 1)
        boolean_closeness_mat = (good_diff < closest_bad_distance)
        close_good_cards = boolean_closeness_mat.sum(dim=1)  # amount of good cards that are closer than the closest
        # bad card - shape (N_vocab)
        close_good_cards[[self.word_index_dict[word] for word in good_words + bad_words]] = 0  # avoid current cards

        # new - choose the best option among the optimal clues ####################
        all_best_indices = torch.nonzero(close_good_cards == close_good_cards.max()).view(-1)
        best_word_cluesize = close_good_cards[all_best_indices[0]]
        # print("amount of optimal clues:", len(all_best_indices))
        best_good_diff = good_diff[all_best_indices]
        closest_bad_distance = closest_bad_distance[all_best_indices]
        best_good_diff = torch.where(best_good_diff < closest_bad_distance,
                                     best_good_diff,
                                     torch.zeros(best_good_diff.shape, dtype=torch.float))
        scores = self.tie_break_scores(best_good_diff, closest_bad_distance.view(-1))
        best_word_idx = torch.argmin(scores)  # index of best word among optimals
        best_word_idx = 0  ############################ TODELETE #####################
        best_word_idx = all_best_indices[best_word_idx]  # that word's index in the entire vocabulary

        clue_word = self.index_word_dict[best_word_idx.item()]

        cluster_words_indices = boolean_closeness_mat[best_word_idx].nonzero(as_tuple=True)[0]
        targets = sorted([good_words[i] for i in cluster_words_indices])
        return clue_word, best_word_cluesize.item(), targets, {'optimal_amount': len(all_best_indices)}


class EmbeddingNearestReceiver(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(EmbeddingNearestReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method,
                                                       dist_metric)

    def forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        clue_word, clue_amount = clue
        clue_array = self.embedder(clue_word)
        words_array = self.embedder(words)
        norm_dif = torch.norm(words_array - clue_array, dim=1)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return {words[i] for i in indices}


class ContextualizedReceiver(rotem_utils.ContextEmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(ContextualizedReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method,
                                                     dist_metric)

    def forward(self, words: List[str], clue: tuple):
        clue_word, clue_amount = clue
        clue_array = self.embedder(clue_word, context=clue_word).view(-1)    # (768)
        assert len(clue_array.shape) == 1
        words_array = self.embedder(words, context=clue_word)       # (N_words, 768)
        norm_dif = torch.norm(words_array - clue_array, dim=1)      # (N_word)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return {words[i] for i in indices}


class CodenameDataLoader:
    def __init__(self, total_cards_amount: int, good_cards_amount: int, embed_method1: str = 'GloVe',
                 embed_method2: str = 'GloVe', agent_vocab: str = 'GloVe', board_vocab: str = 'wordnet-nouns'):
        """
        Note: the embed_methods are only used to filter the vocabulary in this class.
        """
        vocabs = {embed_method1, embed_method2, agent_vocab, board_vocab}  # taking set to avoid duplicates
        self.vocab = rotem_utils.intersect_vocabularies(*vocabs)
        self.vocab_length = len(self.vocab)
        print("length of dataloader vocabulary:", self.vocab_length)
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
            receiver_input = random.sample(words, len(words))  # shuffled words
            yield sender_input, receiver_input

    def legal_word(self, word):
        return word in self.vocab


def skyline_main(tca, gca, sender_type, sender_emb_method, receiver_type, receiver_emb_method, agent_vocab, board_vocab,
                 dist_metric, trial_amount, exhaustive_tiebreak=None, verbose=False):
    """
    run many (trial_amount) experiment games and collect data, for evaluations.

    Parameters
    ----------
    tca - total cards amount
    gca - good cards amount
    sender_type - 'exhaustive' or 'cluster', which sender class to use
    sender_emb_method - 'GloVe' or 'word2vec'
    receiver_type - 'embedding' or 'context'
    receiver_emb_method - 'GloVe' or 'word2vec'
    agent_vocab - 'GloVe' or 'word2vec' or 'nltk-words' or 'wordnet-nouns' or 'nouns-kaggle'
    board_vocab - 'GloVe' or 'word2vec' or 'nltk-words' or 'wordnet-nouns' or 'nouns-kaggle'
    dist_metric - 'l2' or 'cosine_sim'
    trial_amount - how many games to run for the evaluation
    exhaustibe_tiebreak - input to the exhaustive sender
    verbose - bool

    Returns
    -------

    """
    assert sender_type != 'exhaustive' or exhaustive_tiebreak is not None, "exhaustive sender must receive tiebreak"
    kwargs = dict(total_cards=tca, good_cards=gca, vocab_method=agent_vocab, dist_metric=dist_metric)

    sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[sender_type]
    sender_extra_kwargs = {'tie_break_method': exhaustive_tiebreak} if sender_type == 'exhaustive' else {}
    sender = sender_class(**kwargs, **sender_extra_kwargs, embedding_method=sender_emb_method)

    receiver_class = {'embedding': EmbeddingNearestReceiver, 'context': ContextualizedReceiver}[receiver_type]
    receiver = receiver_class(**kwargs, embedding_method=receiver_emb_method)
    dataloader = CodenameDataLoader(tca, gca, sender_emb_method, receiver_emb_method, agent_vocab, board_vocab)
    dataloader = iter(dataloader)

    clue_words = []
    clue_sizes = []
    sender_times = []
    receiver_times = []
    accuracies = []
    optimal_amounts = []

    for _ in tqdm(range(trial_amount)):
        sender_input, receiver_input = next(dataloader)
        t0 = time.perf_counter()
        clue_word, clue_size, targets, extra_data = sender(sender_input, verbose=False)
        t1 = time.perf_counter()
        choice = receiver(receiver_input, (clue_word, clue_size))
        t2 = time.perf_counter()

        sender_time = t1 - t0
        receiver_time = t2 - t1
        accuracy = len([x for x in choice if x in targets]) / clue_size
        optimal_amount = extra_data.get('optimal_amount')

        clue_words.append(clue_word)
        clue_sizes.append(clue_size)
        sender_times.append(sender_time)
        receiver_times.append(receiver_time)
        accuracies.append(accuracy)
        optimal_amounts.append(optimal_amount)

        if verbose:
            # print(f"receiver_input - all words: {receiver_input}\n")
            # print(f"sender input:\n \tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
            # print(f"\tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
            # print(f"target words: {targets} \n \t clue: {clue[0]} \n")
            # print("sender time:", time.perf_counter() - t0)
            # print(f"clue size {clue_size}, guess success: {100 * accuracy}%")
            pass

    if verbose:
        print("experiment results for:")
        print(f"""
        Total cards amount: {tca}
        Good cards amount: {gca}
        Sender type: {sender_type}
        Sender embedding: {sender_emb_method}
        Receiver embedding: {receiver_emb_method}
        Agent vocabulary restriction: {agent_vocab}
        Board vocabulary restriction: {board_vocab}
        Distance/similarity metric: {dist_metric}
        Tiebreak (if exhaustive sender): {exhaustive_tiebreak} \n\n
        Avg. clue size: {sum(clue_sizes) / len(clue_sizes)}
        Avg. receiver accuracy: {sum(accuracies) / len(accuracies)}
        Avg. optimal amount (if exhaustive sender): {sum(optimal_amounts) / len(optimal_amounts)
        if sender_type == 'exhaustive' else None}
        Avg. sender time: {sum(sender_times) / len(sender_times)}
        Avg. receiver time: {sum(receiver_times) / len(receiver_times)}
        """)
    return clue_words, clue_sizes, sender_times, receiver_times, accuracies


def interactive_game():
    defaults = {'tca': 5, 'gca': 2, 'agent_vocab': 'wordnet-nouns', 'board_vocab': 'codenames-words',
                'dist_metric': 'cosine_sim'}
    game1_defaults = {'sender_type': 'exhaustive', 'sender_embedding': 'GloVe', 'tie_break_method': 'red_blue_diff'}
    game2_defaults = {'receiver_type': 'embedding', 'receiver_embedding': 'GloVe'}
    print("This is an interactive codenames game. You have two options: \n"
          "1) Guess words using a clue generated by our sender \n"
          "2) Choose a clue and give it to our receiver \n \n")
    game_choice = rotem_utils.user_choice('1', "Which option do you choose?", options=['1', '2'], full_text=True)
    defaults.update(game1_defaults if game_choice == '1' else game2_defaults)
    print(f"the default parameters are")
    for k, v in defaults.items():
        print(f"\t {k}: \t {v}")
    print('\n')
    defaults_choice = rotem_utils.user_choice('yes', "Use the defaults? (\'yes\' or \'no\')",
                                              options=['yes', 'no'], full_text=True)
    defaults_choice = defaults_choice == 'yes'

    if defaults_choice:
        tca = defaults['tca']
        gca = defaults['gca']
        agent_vocab = defaults['agent_vocab']
        board_vocab = defaults['board_vocab']
        dist_metric = defaults['dist_metric']
    else:
        print("now you choose the game parameters. To take the default, enter \'a\' at any decision. \n")

        tca = rotem_utils.user_choice(defaults['tca'], "total cards amount", demand_int=True)
        gca = rotem_utils.user_choice(defaults['gca'], "good cards amount", demand_int=True)
        agent_vocab = rotem_utils.user_choice(defaults['agent_vocab'], "players' vocabulary limitation",
                                              options=['GloVe', 'word2vec', 'nltk-words',
                                                       'wordnet-nouns', 'nouns-kaggle', 'codenames-words'])
        board_vocab = rotem_utils.user_choice(defaults['board_vocab'], "board vocabulary limitation",
                                              options=['GloVe', 'word2vec', 'nltk-words',
                                                       'wordnet-nouns', 'nouns-kaggle', 'codenames-words'])
        dist_metric = rotem_utils.user_choice(defaults['dist_metric'], "distance metric", options=['l2', 'cosine_sim'])
    kwargs = dict(total_cards=tca, good_cards=gca, vocab_method=agent_vocab, dist_metric=dist_metric)

    if game_choice == '1':
        if defaults_choice:
            sender_type = defaults['sender_type']
            if sender_type == 'exhaustive':
                kwargs['tie_break_method'] = defaults['tie_break_method']
            emb_method = defaults['sender_embedding']
        else:
            sender_type = rotem_utils.user_choice(defaults['sender_type'], "sender type",
                                                  options=['exhaustive', 'cluster'])
            if sender_type == 'exhaustive':
                kwargs['tie_break_method'] = rotem_utils.user_choice(defaults['tie_break_method'],
                                                                     "tie-breaking method",
                                                                     options=['avg_blue_dist', 'max_blue_dist',
                                                                              'max_radius', 'red_blue_diff'])
            emb_method = rotem_utils.user_choice('GloVe', "sender embedding method", options=['GloVe', 'word2vec'])
        dataloader = CodenameDataLoader(tca, gca, emb_method, emb_method, agent_vocab, board_vocab)
        sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[sender_type]
        sender = sender_class(**kwargs, embedding_method=emb_method)

        for sender_input, receiver_input in dataloader:
            print(f"\n words on the board: \n {receiver_input} \n")
            clue, clue_num, targets, _ = sender(sender_input, verbose=False)
            print(f"The sender gives the clue \"{clue}\", for {clue_num} words. Can you guess them?")
            guesses = []
            for i in range(clue_num):
                guess = rotem_utils.user_choice('<stop>', f"guess number {i + 1}:", options=['stop'] + receiver_input,
                                                full_text=True)
                if guess == '<stop>':
                    break
                guesses.append(guess)
            guesses.sort()
            print(f"your guesses: \t {guesses} \n you got right: \t {[w for w in guesses if w in targets]} \n"
                  f"you got wrong: \t {[w for w in guesses if w not in targets]} \n the true words: \t {targets}")
            answer = rotem_utils.user_choice('yes', "play again?", options=['yes', 'no'], full_text=True)
            if answer == 'no':
                break
    else:  # game_choice == '2'
        if defaults_choice:
            emb_method = defaults['receiver_embedding']
            receiver_type = defaults['receiver_type']
        else:
            receiver_type = rotem_utils.user_choice(defaults['receiver_type'], "receiver type",
                                                    options=['embedding', 'context'])
            if receiver_type == 'embedding':
                emb_method = rotem_utils.user_choice('GloVe', "receiver embedding method", options=['GloVe', 'word2vec'])
            else:
                emb_method = 'bert'
        dataloader = CodenameDataLoader(tca, gca, emb_method, emb_method, agent_vocab, board_vocab)
        receiver_class = {'embedding': EmbeddingNearestReceiver, 'context': ContextualizedReceiver}[receiver_type]
        receiver = receiver_class(**kwargs, embedding_method=emb_method)
        for sender_input, receiver_input in dataloader:
            print(f"\tgood words: \t {sender_input[0]}\n \tbad words: \t {sender_input[1]}")
            print("What is your clue? \n")
            while True:
                clue = input()
                if receiver.known_word(clue):
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


def synthetic_sender_round(good_words: list, bad_words: list, sender_type: str, emb_method: str,
                           vocab: str, dist_metric: str, exhaustive_tiebreak: str):
    assert sender_type != 'exhaustive' or exhaustive_tiebreak is not None, "exhaustive sender must receive tiebreak"
    kwargs = dict(total_cards=len(good_words) + len(bad_words), good_cards=len(good_words),
                  vocab_method=vocab, dist_metric=dist_metric)
    sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[sender_type]
    sender_extra_kwargs = {'tie_break_method': exhaustive_tiebreak} if sender_type == 'exhaustive' else {}
    sender = sender_class(**kwargs, **sender_extra_kwargs, embedding_method=emb_method)
    sender_input = (good_words, bad_words)
    clue_word, clue_size, targets, extra_data = sender(sender_input, verbose=False)
    print(f"\t clue: \t {clue_word} \n \t for: \t {targets} \n \t extra data \t {extra_data}")


def synthetic_receiver_round(receiver_type: str, good_words: list, bad_words: list, targets: list, clue: str,
                             emb_method: str, vocab: str, dist_metric: str):
    kwargs = dict(total_cards=len(good_words) + len(bad_words), good_cards=len(good_words),
                  vocab_method=vocab, dist_metric=dist_metric, embedding_method=emb_method)
    receiver_class = {'embedding': EmbeddingNearestReceiver, 'context': ContextualizedReceiver}[receiver_type]
    receiver = receiver_class(**kwargs)
    receiver_input = good_words + bad_words
    choice = receiver(receiver_input, (clue, len(targets)))
    print(f"\t clue: \t {clue} \n \t for: \t {targets} \n \t the receiver chose \t {choice}")


if __name__ == '__main__':
    # clue_words, clue_sizes, sender_times, receiver_times, accuracies = skyline_main(tca=10, gca=3,
    #                                                                                 sender_type='exhaustive',
    #                                                                                 exhaustive_tiebreak='red_blue_diff',
    #                                                                                 sender_emb_method='GloVe',
    #                                                                                 receiver_type='context',
    #                                                                                 receiver_emb_method='bert',
    #                                                                                 agent_vocab='wordnet-nouns',
    #                                                                                 board_vocab='codenames-words',
    #                                                                                 dist_metric='cosine_sim',
    #                                                                                 trial_amount=100, verbose=True)

    interactive_game()

    # model, tokenizer = rotem_utils.initiate_bert()
    # rotem_utils.bert_multiple_context_emb(model, tokenizer, word_list=['dog', 'cat'], context='well')

    # synthetic_sender_round(good_words=['cat', 'dog', 'mouse'],
    #                        bad_words=['electricity', 'stick', 'child'],
    #                        sender_type='exhaustive', emb_method='GloVe', vocab='GloVe',
    #                        dist_metric='cosine_sim', exhaustive_tiebreak='red_blue_diff')

    # synthetic_receiver_round(receiver_type='context', good_words=['cat', 'dog', 'mouse'],
    #                          bad_words=['electricity', 'stick', 'child'],
    #                          clue='family',
    #                          targets=['cat', 'dog'],
    #                          emb_method='bert', vocab='GloVe',
    #                          dist_metric='cosine_sim')

    # model, tokenizer = rotem_utils.initiate_bert()
    # ce_sentences = []
    # BiS_sentences = []
    # keys = [('cat', 'animal'), ('dollar', 'money'), ('monkey', 'banana'),
    #                ('apple', 'eat'), ('star', 'space'), ('radical', 'movement'), ('banana', 'apple')]
    # for w1, w2 in keys:
    #     sent1, _ = rotem_utils.bert_sentence_scorer(model, tokenizer, w1, w2, method='BiS')
    #     sent2, _ = rotem_utils.bert_sentence_scorer(model, tokenizer, w1, w2, method='ce')
    #     BiS_sentences.append(sent1)
    #     ce_sentences.append(sent2)
    # df = pd.DataFrame({'ce': ce_sentences, 'BiS': BiS_sentences}, index=keys)
    # print(df)
