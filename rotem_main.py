import torch
import torch.nn as nn
from torch.nn import functional as F

import rotem_utils
from agents import *
import evaluation_measures as EvalMeasures

import random
import os
from itertools import product
import time
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import math

class CodenameDataLoader:
    def __init__(self, total_cards_amount: int, good_cards_amount: int, agent_vocab: list,
                 board_vocab: str = 'wordnet-nouns'):
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
            yield next(self)

    def __next__(self):
        indices = torch.multinomial(torch.ones(self.vocab_length), self.tca)
        words = [self.vocab[i] for i in indices]
        g_words, b_words = words[:self.gca], words[self.gca:]
        sender_input = (g_words, b_words)
        receiver_input = random.sample(words, len(words))  # shuffled words
        return sender_input, receiver_input

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
    reduction_method: str = 'centroid'  # for cluster sender
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
        str_inputs = ['sender_type', 'exhaustive_tiebreak', 'reduction_method', 'sender_emb_method', 'receiver_type',
                      'receiver_emb_method',
                      'agent_vocab', 'board_vocab', 'dist_metric']
        bool_inputs = ['verbose', 'sender_linear_translation', 'receiver_linear_translation']
        legal_str_options = {'sender_type': ['exhaustive', 'cluster', 'random'],
                             'exhaustive_tiebreak': ['avg_blue_dist', 'max_blue_dist', 'max_radius', 'red_blue_diff'],
                             'reduction_method': ['centroid', 'few-shot-1_token', 'few-shot-greedy', 'few-shot-api',
                                                  'fusion'],
                             'sender_emb_method': ['GloVe', 'word2vec'],
                             'receiver_type': ['embedding', 'context', 'random'],
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
        self.intersected_agent_vocab = rotem_utils.intersect_vocabularies(self.agent_vocab, self.sender_emb_method,
                                                                          self.receiver_emb_method)
        # print(f"intersected_agent_vocab received types {type(self.agent_vocab)}, {type(self.sender_emb_method)},"
        #       f" {type(self.receiver_emb_method)} and values: \n {self.agent_vocab} \n {self.sender_emb_method}"
        #       f" \n {self.receiver_emb_method}")

    def _make_sender(self):
        sender_class = {'exhaustive': ExhaustiveSearchSender,
                        'cluster': EmbeddingClusterSender,
                        'random': CompletelyRandomSender}[self.sender_type]
        kwargs = dict(total_cards=self.tca, good_cards=self.gca, vocab=self.intersected_agent_vocab,
                      dist_metric=self.dist_metric, embedding_method=self.sender_emb_method)
        sender_extra_kwargs = {'exhaustive': {'tie_break_method': self.exhaustive_tiebreak},
                               'cluster': {'reduction_method': self.reduction_method},
                               'random': {}
                               }[self.sender_type]
        self.sender_instance = sender_class(**kwargs, **sender_extra_kwargs)
        if self.sender_linear_translation:
            # receiver_embedding = self.receiver_emb_method if self.receiver_instance is None else self.receiver_instance.embeddings
            receiver_embedding = self.receiver_emb_method
            self.sender_instance.translate_embedding(receiver_embedding)

    def _make_receiver(self):
        receiver_class = {'embedding': EmbeddingNearestReceiver,
                          'context': ContextualizedReceiver,
                          'random': CompletelyRandomReceiver}[self.receiver_type]
        kwargs = dict(total_cards=self.tca, good_cards=self.gca, vocab=self.intersected_agent_vocab,
                      dist_metric=self.dist_metric, embedding_method=self.receiver_emb_method)
        receiver_extra_kwargs = {'embedding': {},
                                 'context': {},
                                 'random': {}
                                 }[self.receiver_type]
        self.receiver_instance = receiver_class(**kwargs, **receiver_extra_kwargs)
        if self.receiver_linear_translation:
            # sender_embedding = self.sender_emb_method if self.sender_instance is None else self.sender_instance.embeddings
            sender_embedding = self.sender_emb_method
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

    def game_instance(self):
        return self.sender, self.receiver, self.dataloader


def skyline_main(trial_amount: int = 50, verbose: bool = False, **opts_kwargs):
    """
    run many (trial_amount) experiment games and collect data, for evaluations.
    """
    opts = CodenamesOptions(**opts_kwargs)
    sender, receiver, dataloader = opts.game_instance()

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
    defaults = {'tca': 6, 'gca': 2, 'agent_vocab': 'wordnet-nouns', 'board_vocab': 'codenames-words',
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
            clue, clue_num, targets, _ = sender(sender_input, verbose=False)
            if tca in [4, 6]:
                EvalMeasures.receiver_API(words=receiver_input, num_choices=len(targets), clue=clue,
                                          targets=targets, gt_good_words=sender_input[0])
            else:
                print(f"\n words on the board: \n {receiver_input} \n")
                print(f"The sender gives the clue \"{clue}\", for {clue_num} words. Can you guess them?")
                guesses = []
                for i in range(clue_num):
                    guess = rotem_utils.user_choice('<stop>', f"guess number {i + 1}:",
                                                    options=['<stop>'] + receiver_input,
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
            if tca % 2 == 0:
                EvalMeasures.plot_given_board(good_words=sender_input[0], bad_words=sender_input[1], grey_words=[])
            else:
                print(f"(can't plot boards with odd number of cards, here it's {tca=})")
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
                max_iter: int = None, deterministic_agents: bool = False, **opts_kwargs):
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
    deterministic_agents
        if the agents are deterministic, there is no need to continue calculating if no progress is made in any single round.

    Returns
    -------
    list of clues, list of guesses, list of targets guessesd and the amount of iterations (which is the length of the other lists)
    """
    sender: EmbeddingAgent
    receiver: EmbeddingAgent
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
                color = 'blue' if guess in good_words else 'red'
                opts.sender.update_knowledge(guess, color)
                opts.receiver.update_knowledge(guess, color)
                break
        good_words = [word for word in good_words if word not in relevant_choice]

        clue_words.append(clue[0])
        guesses.append(choice)
        targets_chosen.append(relevant_choice)
        if len(good_words) == 0:
            break
        if deterministic_agents and len(relevant_choice) == 0:
            clue_words = clue_words + [clue[0]] * (max_iter - len(clue_words))
            guesses = guesses + [choice] * (max_iter - len(guesses))
            targets_chosen = targets_chosen + [[] for __ in range(max_iter - len(guesses))]
            break
    return clue_words, guesses, targets_chosen, len(clue_words)


def multiple_game_metric(N: int, accept_unintended_blue: bool = False,
                         max_iter: int = None, deterministic_agents: bool = False, **opts_kwargs):
    """
    averages the game metric over multiple runs.
    """
    assert 'tca' in opts_kwargs and 'gca' in opts_kwargs, "multiple_game_metric must receive tca and gca"
    opts = CodenamesOptions(**opts_kwargs)
    sender, receiver, dataloader = opts.game_instance()
    new_opts_kwargs = {k: v for k, v in opts_kwargs.items() if
                       k not in ['tca', 'gca', 'sender_instance', 'receiver_instance']}

    metric_sum = 0
    for _ in range(N):
        (good_words, bad_words), _ = next(dataloader)
        sender.reset_knowledge()
        receiver.reset_knowledge()
        *_, metric = game_metric(good_words, bad_words, accept_unintended_blue, max_iter,
                                 deterministic_agents=deterministic_agents,
                                 sender_instance=sender, receiver_instance=receiver, **new_opts_kwargs)
        metric_sum += metric

    return metric_sum / N


def test_glove_w2v(**opts_kwargs):
    avg_metrics = dict()
    combinations = list(product(['word2vec', 'word2vec_T', 'GloVe_T'], ['GloVe', 'GloVe_T', 'word2vec_T']))
    opts_w_g_t = CodenamesOptions(**opts_kwargs,
                                  sender_emb_method="word2vec",
                                  receiver_emb_method="GloVe",
                                  sender_linear_translation=True, receiver_linear_translation=True)
    W_sender_t = opts_w_g_t.sender
    G_receiver_t = opts_w_g_t.receiver
    opts_w_g = CodenamesOptions(**opts_kwargs,
                                sender_emb_method="word2vec",
                                receiver_emb_method="GloVe")
    W_sender = opts_w_g.sender
    G_receiver = opts_w_g.receiver
    opts_g_w_t = CodenamesOptions(**opts_kwargs,
                                  sender_emb_method="GloVe",
                                  receiver_emb_method="word2vec",
                                  sender_linear_translation=True, receiver_linear_translation=True)
    G_sender_t = opts_g_w_t.sender
    W_receiver_t = opts_g_w_t.receiver
    dataloader = iter(opts_w_g_t.dataloader)

    for sender_embedding, receiver_embedding in combinations:
        print(f"calculating score for {sender_embedding} sender and {receiver_embedding} receiver")
        sender = {'word2vec': W_sender, 'word2vec_T': W_sender_t, 'GloVe_T': G_sender_t}[sender_embedding]
        receiver = {'GloVe': G_receiver, 'GloVe_T': G_receiver_t, 'word2vec_T': W_receiver_t}[receiver_embedding]
        metric_array = []
        for _ in tqdm(range(20)):
            (good_words, bad_words), _ = next(dataloader)
            *_, metric = game_metric(good_words=good_words, bad_words=bad_words,
                                     sender_instance=sender, receiver_instance=receiver)
            metric_array.append(metric)
        avg_metrics[(sender_embedding, receiver_embedding)] = sum(metric_array) / len(metric_array)
        print(
            f"for {sender_embedding} sender and {receiver_embedding} receiver: score {sum(metric_array) / len(metric_array)}")
    sender_embeds, receiver_embeds = zip(*combinations)
    metric_score = [avg_metrics[k] for k in combinations]
    df = pd.DataFrame({"Sender": sender_embeds, "Receiver": receiver_embeds, "metric(low=good)": metric_score})
    print(df)


def noise_experiment(**opts_kwargs):
    # word2vec sender, noised word2vec receiver
    noises = torch.logspace(-10, 0.55, 20).tolist()
    embedding_mat_dists = []
    embedding_metric_dists = []
    game_metric_dists = []
    opts = CodenamesOptions(**opts_kwargs, receiver_type='embedding',
                            sender_emb_method="word2vec", receiver_emb_method="word2vec", agent_vocab='word2vec')
    W_sender = opts.sender
    receiver = opts.receiver
    vocab = rotem_utils.intersect_vocabularies('word2vec', 'nouns-kaggle')

    for i, noise in enumerate(noises):
        print(f"calculating results for noise {noise}, number {i + 1} out of {len(noises)}")
        word2vec_embedding_mat = W_sender.embeddings.clone()
        noise = torch.normal(torch.zeros(word2vec_embedding_mat.size()), torch.ones(1) * noise)
        word2vec_embedding_mat = word2vec_embedding_mat + noise
        receiver.embeddings = word2vec_embedding_mat

        embedding_mat_dists.append(torch.norm(W_sender.embeddings - word2vec_embedding_mat).item())
        embedding_metric_dists.append(rotem_utils.average_rank_embedding_distance(W_sender, receiver, vocab=vocab))
        game_metric_dists.append(multiple_game_metric(N=5, accept_unintended_blue=False, deterministic_agents=False,
                                                      sender_emb_method="word2vec", receiver_emb_method="word2vec",
                                                      agent_vocab='word2vec', sender_instance=W_sender,
                                                      receiver_instance=receiver, **opts_kwargs))
    df = pd.DataFrame({'noise': noises, 'norm_difference': embedding_mat_dists,
                       'metric_difference': embedding_metric_dists, 'game_difference': game_metric_dists})
    print(df)
    if not os.path.isdir(rotem_utils.ROTEM_RESULTS_DIR):
        os.mkdir(rotem_utils.ROTEM_RESULTS_DIR)
    df.to_csv(os.path.join(rotem_utils.ROTEM_RESULTS_DIR, 'noise_experiment_26_07'))


if __name__ == '__main__':
    # clue_words, clue_sizes, sender_times, receiver_times, accuracies = skyline_main(trial_amount=50, verbose=True,
    #                                                                                 tca=6, gca=3,
    #                                                                                 sender_type='cluster',
    #                                                                                 reduction_method='few-shot-greedy',
    #                                                                                 sender_emb_method='word2vec',
    #                                                                                 receiver_type='embedding',
    #                                                                                 receiver_emb_method='word2vec',
    #                                                                                 board_vocab='codenames-words',
    #                                                                                 dist_metric='cosine_sim'
    #                                                                                 )

    # noise_experiment(tca=20, gca=10)
    # rotem_utils.noise_experiment_plots()

    # interactive_game()

    # model, tokenizer = rotem_utils.initiate_bert()
    # rotem_utils.bert_multiple_context_emb(model, tokenizer, word_list=['dog', 'cat'], context='well')

    # opts = CodenamesOptions(sender_type='cluster', reduction_method='few-shot-greedy',
    #                         sender_emb_method='word2vec', agent_vocab='codenames-words',
    #                         receiver_emb_method='word2vec', dist_metric='cosine_sim',
    #                         tca=6, gca=3)
    # with open('few-shot_clues_gen.txt', 'w') as writefile:
    #     for _ in range(30):
    #         sender_input, _ = next(opts.dataloader)
    #         clue_word, clue_size, targets, extra_data = opts.sender(sender_input, verbose=False)
    #         # print('good words:', good_words)
    #         # print('bad words:', bad_words)
    #         print(f"\t clue: \t {clue_word} \n \t for: \t {targets} \n")
    #         writefile.write(f"\t clue: \t {clue_word} \n \t for: \t {targets} \n \n")

    opts = CodenamesOptions(receiver_type='embedding',
                            sender_emb_method='word2vec', agent_vocab='codenames-words',
                            receiver_emb_method='word2vec', dist_metric='cosine_sim',
                            tca=20, gca=6)
    sender, receiver, dataloader = opts.game_instance()
    clues = []
    for _ in tqdm(list(range(30000))):
        sender_input, _ = next(dataloader)
        clue_word, clue_size, targets, extra_data = sender(sender_input, verbose=False)
        clues.append((clue_word, clue_size))
    _, receiver_input = next(dataloader)
    t0 = time.perf_counter()
    choices1 = receiver.faiss_forward(receiver_input, clues)
    t1 = time.perf_counter()
    choices2 = [receiver(receiver_input, clue) for clue in clues]
    t2 = time.perf_counter()
    print(f"{choices1[0]=}, {choices2[0]=}")
    print(f"time with faiss: {t1 - t0}, time with list comp: {t2 - t1}")

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

    # opts1 = CodenamesOptions(sender_emb_method='GloVe', receiver_emb_method='word2vec', sender_linear_translation=True,
    #                          tca=20, gca=10)
    # sender = opts1.sender
    # opts2 = CodenamesOptions(sender_emb_method='word2vec', receiver_emb_method='GloVe', receiver_linear_translation=True)
    # receiver = opts2.receiver
    # print("equal?", sender.vocab == receiver.vocab)
    # (good_words, bad_words), all_words = next(iter(opts1.dataloader))
    # *_, metric = game_metric(good_words=good_words, bad_words=bad_words, max_iter=100,
    #                          sender_instance=sender, receiver_instance=receiver)
    # print("metric:", metric)

    # tca = 20
    # gca = 10
    # vocab = rotem_utils.intersect_vocabularies('GloVe', 'word2vec', 'nouns-kaggle')
    # print("vocab size:", len(vocab))
    # agent_w2v = rotem_utils.EmbeddingAgent(tca, gca, 'word2vec', vocab=vocab)
    # agent_w2v_p = rotem_utils.EmbeddingAgent(tca, gca, 'word2vec', vocab=vocab)
    #
    # word2vec_embedding_mat = agent_w2v_p.embeddings
    # noise = torch.rand(word2vec_embedding_mat.size(), dtype=word2vec_embedding_mat.dtype)
    # noise /= math.sqrt(word2vec_embedding_mat.shape[1])
    # agent_w2v_p.embeddings = word2vec_embedding_mat + noise
    #
    # agent_glove = rotem_utils.EmbeddingAgent(tca, gca, 'GloVe', vocab=vocab)
    # agent_glove_translated = rotem_utils.EmbeddingAgent(tca, gca, 'GloVe', vocab=vocab, translate_to='word2vec')
    #
    # d1 = rotem_utils.average_rank_embedding_distance(agent_w2v, agent_w2v_p)
    # d2 = rotem_utils.average_rank_embedding_distance(agent_w2v, agent_glove)
    # d3 = rotem_utils.average_rank_embedding_distance(agent_w2v, agent_glove_translated)
    # print(f"distance between word2vec and word2vec perturbed: {d1}")
    # print(f"distance between word2vec and GloVe: {d2}")
    # print(f"distance between word2vec and GloVe translated: {d3}")
