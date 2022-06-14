import torch
import torch.nn as nn
from torch.nn import functional as F

import rotem_utils

import random
from itertools import product
from collections import defaultdict
from typing import List
from scipy.cluster.vq import kmeans2
import time
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd


class EmbeddingClusterSender(rotem_utils.EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str,
                 reduction_method: str = 'centroid'):
        super(EmbeddingClusterSender, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)
        cluster_range = min(10, good_cards)
        multiplicity = 8
        self.cluster_amounts = sum([[self.tca // (i + 1)] * multiplicity for i in range(cluster_range)], start=[])
        self.good_embeds = None
        self.bad_embeds = None
        assert reduction_method in ['centroid', 'few-shot', 'fusion']
        self.reduction_method = reduction_method

    def reduce_cluster(self, centroid, indices, targets, words) -> str:
        """
        given a cluster of good words, find a word that describes the cluster using the reduction method for __init__
        Parameters
        ----------
        centroid
            l2 mean of embeddings
        indices
            the indices of the word in good_words and self.good_embeds
        targets
            actual string words of the cluster
        words
            all words on the board

        Returns
        -------
        chosen word (str)

        """
        if self.reduction_method == 'centroid':
            norm_dif = torch.norm(self.embeddings[:-1] - centroid, dim=1)
            norm_dif[[self.word_index_dict[word] for word in words]] = torch.inf
            index = torch.argmin(norm_dif).item()
            closest_word = self.index_word_dict[index]
            return closest_word
        if self.reduction_method == 'few-shot':
            delimiter = "###"
            prompt = f"group: \"cat\", \"dog\", \"mouse\". description: \"animals\"" \
                     f"{delimiter}" \
                     f"group: \"pants\", \"shirt\", \"hat\". description: \"cloths\"" \
                     f"{delimiter}" \
                     f"group: \"one\", \"two\", \"three\", \"four\". description: \"numbers\"" \
                     f"{delimiter}" \
                     f"group: " + ', '.join(["\"" + t + "\"" for t in targets]) + ". description: "
            # print("prompt: \n" + '-'*12)
            # print(prompt, "\n \n")
            closest_word = rotem_utils.few_shot_huggingface(prompt, delimiter)
            # print("few shot result:", "\n", closest_word)
            return closest_word
        if self.reduction_method == 'fusion':
            # list_length is a hyper-parameter
            list_length = 20
            word_lists = []
            for word_vec in self.good_embeds[indices]:
                norm_dif = torch.norm(self.embeddings[:-1] - word_vec, dim=1)
                norm_dif[[self.word_index_dict[word] for word in words]] = torch.inf
                _, close_words_indices = torch.topk(norm_dif, list_length, largest=False)
                word_lists.append([self.index_word_dict[index] for index in close_words_indices.tolist()])
            scores_dict = defaultdict(int)
            best_val = 0
            arg_best = None
            for word_list in word_lists:
                for i, word in enumerate(word_list):
                    scores_dict[word] += list_length - i
                    if scores_dict[word] > best_val or arg_best is None:
                        best_val = scores_dict[word]
                        arg_best = word
            return arg_best
        raise ValueError("illegal reduction method " + self.reduction_method)

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
        targets = [good_words[i] for i in indices]
        clue_word = self.reduce_cluster(centroid, indices, targets, good_words + bad_words)

        return clue_word, clue_len, sorted(targets), {}


class ExhaustiveSearchSender(rotem_utils.EmbeddingAgent):
    """
    A sender agent which operates as follows - given good and bad cards, finds the set of all words in the vocabulary
    with maximal amount of good cards closer than all bad cards. Out of this set, the agent chosses the word which is
    closest to it's matching good cards.
    """

    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str,
                 tie_break_method: str = 'avg_blue_dist'):
        super(ExhaustiveSearchSender, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)
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
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str):
        super(EmbeddingNearestReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)

    def forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        clue_word, clue_amount = clue
        clue_array = self.embedder(clue_word)
        words_array = self.embedder(words)
        norm_dif = torch.norm(words_array - clue_array, dim=1)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return [words[i] for i in indices]


class ContextualizedReceiver(rotem_utils.ContextEmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab_method: str, dist_metric: str):
        super(ContextualizedReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab_method,
                                                     dist_metric)

    def forward(self, words: List[str], clue: tuple):
        clue_word, clue_amount = clue
        clue_array = self.embedder(clue_word, context=clue_word).view(-1)  # (768)
        assert len(clue_array.shape) == 1
        words_array = self.embedder(words, context=clue_word)  # (N_words, 768)
        norm_dif = torch.norm(words_array - clue_array, dim=1)  # (N_word)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return [words[i] for i in indices]


class CodenameDataLoader:
    def __init__(self, total_cards_amount: int, good_cards_amount: int, agent_vocab: list, board_vocab: str = 'wordnet-nouns'):
        """
        Note: the embed_methods are only used to filter the vocabulary in this class.
        """
        self.vocab = list(set(agent_vocab).intersection(rotem_utils.get_vocabulary(board_vocab)))
        self.vocab_length = len(self.vocab)
        print("length of dataloader vocabulary:", self.vocab_length)
        self.gca = good_cards_amount
        self.tca = total_cards_amount

    def __iter__(self):
        while True:
            indices = torch.multinomial(torch.ones(self.vocab_length), self.tca)
            words = [self.vocab[i] for i in indices]
            g_words, b_words = words[:self.gca], words[self.gca:]
            sender_input = (g_words, b_words)
            receiver_input = random.sample(words, len(words))  # shuffled words
            yield sender_input, receiver_input

    def legal_word(self, word):
        return word in self.vocab


@dataclass
class CodenamesOptions:
    """
    This class holds the game information - amounts of cards, embeddings, vocabularies and other kwargs.
    It generates the agents and the dataloader.

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
    exhaustibe_tiebreak - input to the exhaustive sender
    """
    tca: int = 10
    gca: int = 3
    sender_type: str = 'exhaustive'
    exhaustive_tiebreak: str = 'red_blue_diff'  # for exhaustive sender
    reduction_method: str = 'centroid'          # for cluster sender
    sender_emb_method: str = 'GloVe'
    sender_linear_translation: bool = False
    receiver_type: str = 'embedding'
    receiver_emb_method: str = 'GloVe'
    receiver_linear_translation: bool = False
    agent_vocab: str = 'wordnet-nouns'
    board_vocab: str = 'codenames-words'
    dist_metric: str = 'cosine_sim'
    verbose: bool = False
    sender_instance: rotem_utils.EmbeddingAgent = None
    receiver_instance: rotem_utils.EmbeddingAgent or rotem_utils.ContextEmbeddingAgent = None
    dataloader_instance: CodenameDataLoader = None

    def __post_init__(self):
        # assertions - type and legal choice checking
        int_inputs = ['tca', 'gca']
        str_inputs = ['sender_type', 'exhaustive_tiebreak', 'reduction_method', 'sender_emb_method', 'receiver_type', 'receiver_emb_method',
                      'agent_vocab', 'board_vocab', 'dist_metric']
        bool_inputs = ['verbose', 'sender_linear_translation', 'receiver_linear_translation']
        legal_str_options = {'sender_type': ['exhaustive', 'cluster'],
                             'exhaustive_tiebreak': ['avg_blue_dist', 'max_blue_dist', 'max_radius', 'red_blue_diff'],
                             'reduction_method': ['centroid', 'few-shot', 'fusion'],
                             'sender_emb_method': ['GloVe', 'word2vec'],
                             'receiver_type': ['embedding', 'context'],
                             'receiver_emb_method': ['GloVe', 'word2vec', 'bert', 'deberta'],
                             'agent_vocab': ['GloVe', 'word2vec', 'nltk-words', 'wordnet-nouns',
                                             'nouns-kaggle', 'codenames-words'],
                             'board_vocab': ['GloVe', 'word2vec', 'nltk-words', 'wordnet-nouns',
                                             'nouns-kaggle', 'codenames-words'],
                             'dist_metric': ['cosine_sim', 'l2']}
        attr_values = vars(self)
        for attr_list, attr_type in [(int_inputs, int), (str_inputs, str), (bool_inputs, bool)]:
            for attr in attr_list:
                assert isinstance(attr_values[attr], attr_type), f"{attr} input must be of type {attr_type}"
                if attr_type is str:
                    assert attr_values[attr] in legal_str_options[attr], \
                        f"choose {attr} from {legal_str_options[attr]}. Not an option: {attr_values[attr]}"

        # find common vocabulary for Sender and Receiver
        self.intersected_agent_vocab = rotem_utils.intersect_vocabularies(self.agent_vocab, self.sender_emb_method, self.receiver_emb_method)

    def _make_sender(self):
        sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[self.sender_type]
        kwargs = dict(total_cards=self.tca, good_cards=self.gca, vocab=self.intersected_agent_vocab,
                      dist_metric=self.dist_metric, embedding_method=self.sender_emb_method)
        sender_extra_kwargs = {'exhaustive': {'tie_break_method': self.exhaustive_tiebreak},
                               'cluster': {'reduction_method': self.reduction_method}
                               }[self.sender_type]
        self.sender_instance = sender_class(**kwargs, **sender_extra_kwargs)
        if self.sender_linear_translation:
            receiver_embedding = self.receiver_emb_method if self.receiver_instance is None else self.receiver_instance.embeddings
            self.sender_instance.translate_embedding(receiver_embedding)

    def _make_receiver(self):
        receiver_class = {'embedding': EmbeddingNearestReceiver, 'context': ContextualizedReceiver}[self.receiver_type]
        kwargs = dict(total_cards=self.tca, good_cards=self.gca, vocab=self.intersected_agent_vocab,
                      dist_metric=self.dist_metric, embedding_method=self.receiver_emb_method)
        receiver_extra_kwargs = {'embedding': {},
                                 'context': {}
                                 }[self.receiver_type]
        self.receiver_instance = receiver_class(**kwargs, **receiver_extra_kwargs)
        if self.receiver_linear_translation:
            sender_embedding = self.sender_emb_method if self.sender_instance is None else self.sender_instance.embeddings
            self.receiver_instance.translate_embedding(sender_embedding)

    def _make_dataloader(self):
        dataloader = CodenameDataLoader(self.tca, self.gca, self.intersected_agent_vocab, self.board_vocab)
        self.dataloader_instance = dataloader

    @property
    def sender(self):
        if self.sender_instance is None:
            self._make_sender()
        return self.sender_instance

    @property
    def receiver(self):
        if self.receiver_instance is None:
            self._make_receiver()
        return self.receiver_instance

    @property
    def dataloader(self):
        if self.dataloader_instance is None:
            self._make_dataloader()
        return self.dataloader_instance

    def game_instance(self, iter_dataloader: bool = False):
        dataloader = self.dataloader
        if iter_dataloader:
            dataloader = iter(dataloader)
        return self.sender, self.receiver, dataloader


def skyline_main(opts: CodenamesOptions, trial_amount: int = 50, verbose: bool = False):
    """
    run many (trial_amount) experiment games and collect data, for evaluations.
    """
    sender, receiver, dataloader = opts.game_instance(iter_dataloader=True)

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
        print("experiment results for:")
        print(f"""
        Total cards amount: {opts.tca}
        Good cards amount: {opts.gca}
        Sender type: {opts.sender_type}
        Sender embedding: {opts.sender_emb_method}
        Receiver embedding: {opts.receiver_emb_method}
        Agent vocabulary restriction: {opts.agent_vocab}
        Board vocabulary restriction: {opts.board_vocab}
        Distance/similarity metric: {opts.dist_metric}
        Tiebreak (if exhaustive sender): {opts.exhaustive_tiebreak} \n\n
        Avg. clue size: {sum(clue_sizes) / len(clue_sizes)}
        Avg. receiver accuracy: {sum(accuracies) / len(accuracies)}
        Avg. optimal amount (if exhaustive sender): {sum(optimal_amounts) / len(optimal_amounts)
        if opts.sender_type == 'exhaustive' else None}
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
    kwargs = dict(total_cards=tca, good_cards=gca, dist_metric=dist_metric)

    if game_choice == '1':
        if defaults_choice:
            sender_type = defaults['sender_type']
            if sender_type == 'exhaustive':
                kwargs['tie_break_method'] = defaults['tie_break_method']
            emb_method = defaults['sender_embedding']
            translation_bool, to_embedding = False, ''
        else:
            sender_type = rotem_utils.user_choice(defaults['sender_type'], "sender type",
                                                  options=['exhaustive', 'cluster'])
            if sender_type == 'exhaustive':
                kwargs['tie_break_method'] = rotem_utils.user_choice(defaults['tie_break_method'],
                                                                     "tie-breaking method",
                                                                     options=['avg_blue_dist', 'max_blue_dist',
                                                                              'max_radius', 'red_blue_diff'])
            elif sender_type == 'cluster':
                kwargs['reduction_method'] = rotem_utils.user_choice('centroid', "cluster reduction method",
                                                                     options=['centroid', 'few-shot', 'fusion'])
            emb_method = rotem_utils.user_choice('GloVe', "sender embedding method", options=['GloVe', 'word2vec'])
            translation_bool = rotem_utils.user_choice('no', "Use translation? (\'yes\' or \'no\')",
                                                       options=['yes', 'no'], full_text=True)
            translation_bool = translation_bool == 'yes'
            if translation_bool:
                to_embedding = rotem_utils.user_choice('GloVe', "translate to which embedding?",
                                                       options=['GloVe', 'word2vec'], full_text=True)
        if translation_bool:
            intersected_vocab = rotem_utils.intersect_vocabularies(agent_vocab, emb_method, to_embedding)
        else:
            intersected_vocab = rotem_utils.intersect_vocabularies(agent_vocab, emb_method)
        kwargs['vocab'] = intersected_vocab
        dataloader = CodenameDataLoader(tca, gca, intersected_vocab, board_vocab)
        sender_class = {'exhaustive': ExhaustiveSearchSender, 'cluster': EmbeddingClusterSender}[sender_type]
        sender = sender_class(**kwargs, embedding_method=emb_method)
        if translation_bool:
            sender.translate_embedding(to_embedding)

        for sender_input, receiver_input in dataloader:
            print(f"\n words on the board: \n {receiver_input} \n")
            clue, clue_num, targets, _ = sender(sender_input, verbose=False)
            print(f"The sender gives the clue \"{clue}\", for {clue_num} words. Can you guess them?")
            guesses = []
            for i in range(clue_num):
                guess = rotem_utils.user_choice('<stop>', f"guess number {i + 1}:", options=['<stop>'] + receiver_input,
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
            translation_bool, to_embedding = False, ''
        else:
            receiver_type = rotem_utils.user_choice(defaults['receiver_type'], "receiver type",
                                                    options=['embedding', 'context'])
            if receiver_type == 'embedding':
                emb_method = rotem_utils.user_choice('GloVe', "receiver embedding method",
                                                     options=['GloVe', 'word2vec'])
            else:
                emb_method = rotem_utils.user_choice('bert', "receiver embedding method",
                                                     options=['bert', 'deberta'])
            translation_bool = rotem_utils.user_choice('no', "Use translation? (\'yes\' or \'no\')",
                                                       options=['yes', 'no'], full_text=True)
            translation_bool = translation_bool == 'yes'
            if translation_bool:
                to_embedding = rotem_utils.user_choice('GloVe', "translate to which embedding?",
                                                       options=['GloVe', 'word2vec'], full_text=True)
        if translation_bool:
            intersected_vocab = rotem_utils.intersect_vocabularies(agent_vocab, emb_method, to_embedding)
        else:
            intersected_vocab = rotem_utils.intersect_vocabularies(agent_vocab, emb_method)
        kwargs['vocab'] = intersected_vocab
        dataloader = CodenameDataLoader(tca, gca, intersected_vocab, board_vocab)
        receiver_class = {'embedding': EmbeddingNearestReceiver, 'context': ContextualizedReceiver}[receiver_type]
        receiver = receiver_class(**kwargs, embedding_method=emb_method)
        if translation_bool:
            receiver.translate_embedding(to_embedding)
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


def synthetic_sender_round(good_words: list, bad_words: list, **opts_kwargs):
    opts = CodenamesOptions(**opts_kwargs, tca=len(good_words) + len(bad_words), gca=len(good_words))
    sender_input = (good_words, bad_words)
    clue_word, clue_size, targets, extra_data = opts.sender(sender_input, verbose=False)
    print(f"\t clue: \t {clue_word} \n \t for: \t {targets} \n \t extra data \t {extra_data}")


def synthetic_receiver_round(good_words: list, bad_words: list, targets: list, clue: str, **opts_kwargs):
    opts = CodenamesOptions(**opts_kwargs, tca=len(good_words) + len(bad_words), gca=len(good_words))
    receiver_input = good_words + bad_words
    choice = opts.receiver(receiver_input, (clue, len(targets)))
    print(f"\t clue: \t {clue} \n \t for: \t {targets} \n \t the receiver chose \t {choice}")


def game_metric(good_words: list, bad_words: list, accept_unintended_blue: bool = False,
                max_iter: int = None, **opts_kwargs):
    """
    operates a single game until the receiver guesses all the good words or max_iter is reached.
    Parameters
    ----------
    opts
        the opts object containing the sender and receiver
    good_words
        list of strings
    bad_words
        list of strings
    accept_unintended_blue
        if True, when Receiver guesses a blue card that wasn't a target, it counts for the game.
    max_iter
        bound on iterations (amount of clues) - defaults to len(good_words) + 1

    Returns
    -------
    list of clues, list of guesses, list of targets guessesd and the amount of iterations (which is the length of the other lists)
    """
    opts = CodenamesOptions(**opts_kwargs, tca=len(good_words) + len(bad_words), gca=len(good_words))
    if max_iter is None:
        max_iter = len(good_words) + 1
    clue_words = []
    guesses = []
    targets_chosen = []
    for _ in range(max_iter):
        receiver_input = random.sample(good_words + bad_words, len(good_words + bad_words))  # shuffled words
        *clue, targets, _ = opts.sender((good_words, bad_words), verbose=False)
        choice = opts.receiver(receiver_input, clue)
        reference = good_words if accept_unintended_blue else targets
        relevant_choice = []
        for guess in choice:
            if guess in reference:
                relevant_choice.append(guess)
            else:
                break
        good_words = [word for word in good_words if word not in relevant_choice]

        clue_words.append(clue[0])
        guesses.append(choice)
        targets_chosen.append(relevant_choice)
        if len(good_words) == 0:
            break
    return clue_words, guesses, targets_chosen, len(clue_words)


def test_glove_w2v(tca=100, gca=40, sender_type='exhaustive', receiver_type='embedding', agent_vocab='codenames-words',
                   board_vocab='codenames-words', dist_metric='cosine_sim', exhaustive_tiebreak='red_blue_diff'):
    avg_metrics = dict()
    combinations = list(product(['GloVe', 'word2vec'], ['GloVe', 'word2vec']))
    opts_g_w = CodenamesOptions(tca=tca, gca=gca, sender_type=sender_type, exhaustive_tiebreak=exhaustive_tiebreak,
                                sender_emb_method="GloVe", receiver_type=receiver_type, receiver_emb_method="word2vec",
                                agent_vocab=agent_vocab, board_vocab=board_vocab, dist_metric=dist_metric,
                                sender_linear_translation=True, receiver_linear_translation=True)
    G_sender = opts_g_w.sender
    W_receiver = opts_g_w.receiver
    dataloader = iter(opts_g_w.dataloader)
    opts_w_g = CodenamesOptions(tca=tca, gca=gca, sender_type=sender_type, exhaustive_tiebreak=exhaustive_tiebreak,
                                sender_emb_method="word2vec", receiver_type=receiver_type, receiver_emb_method="GloVe",
                                agent_vocab=agent_vocab, board_vocab=board_vocab, dist_metric=dist_metric,
                                sender_linear_translation=True, receiver_linear_translation=True)
    W_sender = opts_w_g.sender
    G_receiver = opts_w_g.receiver

    for sender_embedding, receiver_embedding in combinations:
        print(f"calculating score for {sender_embedding} sender and {receiver_embedding} receiver")
        sender = {'GloVe': G_sender, 'word2vec': W_sender}[sender_embedding]
        receiver = {'GloVe': G_receiver, 'word2vec': W_receiver}[receiver_embedding]
        metric_array = []
        for _ in tqdm(range(20)):
            (good_words, bad_words), _ = next(dataloader)
            *_, metric = game_metric(good_words, bad_words, sender_instance=sender, receiver_instance=receiver)
            metric_array.append(metric)
        avg_metrics[(sender_embedding, receiver_embedding)] = sum(metric_array) / len(metric_array)
        print(f"for {sender_embedding} sender and {receiver_embedding} receiver: score {sum(metric_array) / len(metric_array)}")
    sender_embeds, receiver_embeds = zip(*combinations)
    metric_score = [avg_metrics[k] for k in combinations]
    df = pd.DataFrame({"Sender": sender_embeds, "Receiver": receiver_embeds, "metric(low=good)": metric_score})
    print(df)


if __name__ == '__main__':
    # opts = CodenamesOptions(tca=10, gca=3,
    #                         sender_type='exhaustive',
    #                         exhaustive_tiebreak='red_blue_diff',
    #                         sender_emb_method='GloVe',
    #                         receiver_type='embedding',
    #                         receiver_emb_method='GloVe',
    #                         agent_vocab='wordnet-nouns',
    #                         board_vocab='codenames-words',
    #                         dist_metric='cosine_sim')
    # clue_words, clue_sizes, sender_times, receiver_times, accuracies = skyline_main(opts, trial_amount=50, verbose=True)

    test_glove_w2v(15, 5, agent_vocab='wordnet-nouns')

    # interactive_game()

    # model, tokenizer = rotem_utils.initiate_bert()
    # rotem_utils.bert_multiple_context_emb(model, tokenizer, word_list=['dog', 'cat'], context='well')

    # synthetic_sender_round(good_words=['tomato', 'cucumber', 'carrot'],
    #                        bad_words=['cat', 'dog', 'mouse'],
    #                        sender_type='exhaustive', sender_emb_method='GloVe', agent_vocab='nouns-kaggle',
    #                        dist_metric='cosine_sim', sender_linear_translation=True, receiver_emb_method='word2vec')

    # synthetic_receiver_round(good_words=['cat', 'dog', 'mouse'],
    #                          bad_words=['electricity', 'stick', 'child'],
    #                          clue='family',
    #                          targets=['cat', 'dog'],
    #                          receiver_type='embedding',
    #                          receiver_emb_method='word2vec', agent_vocab='word2vec',
    #                          dist_metric='cosine_sim', receiver_linear_translation=True)

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

    # delimiter = "###"
    # prompt = f"group: cat, dog, mouse. description: \"animals\" {delimiter} group:" \
    #          f" pants, shirt, hat. description: \"cloths\" {delimiter} group: one, two, three. description: "
    # print(rotem_utils.few_shot_huggingface(prompt, delimiter))
