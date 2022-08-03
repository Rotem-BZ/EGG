import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertForMaskedLM, DebertaTokenizer, DebertaForMaskedLM
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from scipy.cluster.vq import kmeans2
import numpy as np
import faiss

from typing import List
from collections import defaultdict
import random
from os.path import isfile
from abc import ABC

import rotem_utils

pretrained_LM: tuple = ()  # Can be a single BERT instance used by all the agents instead of multiple BERTs
pretrained_fewshot: tuple = ()

GPT_NEO_NAME = "EleutherAI/gpt-neo-2.7B" if isfile('remote_machine_file') else "EleutherAI/gpt-neo-125M"


class Agent(nn.Module, ABC):
    def __init__(self, total_cards_amount: int, good_cards_amount: int):
        super(Agent, self).__init__()
        self.gca = good_cards_amount
        self.tca = total_cards_amount
        self.known_red_cards = []
        self.known_blue_cards = []

    def update_knowledge(self, word: str, color: str):
        if color == 'blue':
            self.known_blue_cards.append(word)
        elif color == 'red':
            self.known_red_cards.append(word)
        else:
            raise ValueError(f"illegal color {color}")

    def reset_knowledge(self):
        self.known_red_cards = []
        self.known_blue_cards = []

    def sender_knowledge_forward(self, words):
        good_words, bad_words = words
        good_words = [x for x in good_words if x not in self.known_blue_cards]
        bad_words = [x for x in bad_words if x not in self.known_red_cards]
        return_tuple = None
        if len(good_words) == 0:
            # shouldn't happen
            print("WARNING good words list is empty!")
        if len(bad_words) == 0:
            # the receiver discovered all red words so no further hinting is required. choose arbitrary clue
            # for the rest of the blue words
            clue_word = None
            for word in self.vocab:
                if word not in good_words + bad_words + self.known_blue_cards + self.known_red_cards:
                    clue_word = word
                    break
            return_tuple = (clue_word, len(good_words), good_words, {'optimal_amount': len(good_words)})
        return good_words, bad_words, return_tuple

    def receiver_knowledge_forward(self, words, clue):
        clue_word, clue_amount = clue
        board_blue_cards = [word for word in words if word in self.known_blue_cards]
        if board_blue_cards:
            clue_amount -= len(board_blue_cards)
        words = [x for x in words if x not in self.known_red_cards + self.known_blue_cards]
        return words, clue_word, clue_amount, board_blue_cards


class EmbeddingAgent(Agent):
    """
    base class for non-contextualized embedding agents.
    Handels the creation of the embedding matrix, intersection of vocabularies if given vocab is str, and defines
    the `embedder` function to turn str (or List[str]) to word embedding(s).
    """

    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'GloVe',
                 vocab: list = None, dist_metric: str = 'cosine_sim', translate_to: str = None):
        super(EmbeddingAgent, self).__init__(total_cards_amount, good_cards_amount)
        self.embeddings, self.word_index_dict, self.index_word_dict = rotem_utils.get_embedder_list_vocab(
            embedding_method, vocab)
        self.vocab = vocab
        self.embedding_method = embedding_method
        self.dist_metric = dist_metric
        assert dist_metric in ['cosine_sim', 'l2']
        if dist_metric == 'cosine_sim':
            # Normalize the embedding vectors. Euclidean norm is now rank-equivalent to cosine similarity, so no
            # further change is needed.
            embedding_norms = torch.norm(self.embeddings, dim=-1, keepdim=True)
            self.embeddings = self.embeddings / embedding_norms
        if translate_to is not None:
            self.translate_embedding(translate_to)

    def embedder(self, word: List[str] or str):
        if isinstance(word, str):
            return self.embeddings[self.word_index_dict.get(word, -1)]
        elif isinstance(word, list):
            indices = [self.word_index_dict.get(w, -1) for w in word]
            return self.embeddings[indices]
        else:
            raise ValueError

    def known_word(self, word: str):
        return word in self.vocab

    def translate_embedding(self, other_embedding, subvocab: str or list = None):
        # If other_embedding is a tesor, it must use the same vocabulary in the same order as self.embeddings
        # If subvobac is a list, self.vocab must inlude all if its words.
        if subvocab is None:
            vocab = self.vocab
            this_embedding = self.embeddings
        else:
            vocab = list(set(rotem_utils.get_vocabulary(subvocab)).intersection(self.vocab)) if isinstance(subvocab, str) else subvocab
            this_embedding = self.embeddings[[self.word_index_dict[w] for w in vocab]]

        if isinstance(other_embedding, str):
            if other_embedding == self.embedding_method:
                return
            other_array, *_ = rotem_utils.get_embedder_list_vocab(other_embedding, vocab)
        elif isinstance(other_embedding, torch.Tensor):
            assert other_embedding.shape[0] == len(vocab), f"{other_embedding.shape[0]=}, {len(vocab)=}"
            other_array = other_embedding.clone().detach()
        else:
            raise ValueError(f"illegal type {type(other_embedding)}, should be str or Tensor")
        # option 1 - closed LS solution
        T = torch.linalg.lstsq(this_embedding, other_array)[0]
        self.embeddings = self.embeddings @ T
        ###
        # # option 2 - learnable MLP
        # T = self.learn_translation_matrix(this_embedding, other_array)
        # self.embeddings = T(self.embeddings).detach()
        ###
        if self.dist_metric == 'cosine_sim':
            embedding_norms = torch.norm(self.embeddings, dim=-1, keepdim=True)
            self.embeddings = self.embeddings / embedding_norms
        print(f"{len(vocab)=}, embed_dimension={self.embeddings.shape[1]}, norm diff: {torch.norm(self.embeddings[[self.word_index_dict[w] for w in vocab]] - other_array)}")


    @staticmethod
    def learn_translation_matrix(f_embedding: torch.Tensor, t_embedding: torch.Tensor, lr=0.001, epochs=15):
        # learn translation matrix by gradient descent
        n, d1 = f_embedding.shape
        _, d2 = t_embedding.shape

        # translation_mat = nn.Linear(d1, d2, bias=False)
        translation_mat = nn.Sequential(
            nn.Linear(d1, 300, bias=False),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, d2)
        )
        optimizer = torch.optim.SGD(translation_mat.parameters(), lr=lr)
        batch_size = 40
        losses = defaultdict(list)
        for epoch in range(epochs):
            if epoch == epochs // 2:
                batch_size = 100
            print(f"epoch {epoch} / {epochs}")
            optimizer.zero_grad()
            for i in range(n // batch_size):
                from_vectors = f_embedding[batch_size * i: batch_size * (i + 1)].clone()
                to_vectors = t_embedding[batch_size * i: batch_size * (i + 1)].clone()
                output = translation_mat(from_vectors)
                loss = torch.norm(output - to_vectors)
                loss /= from_vectors.shape[0]
                loss.backward()
                losses[f"epoch_{epoch}"].append(loss.item())
                optimizer.step()
        return translation_mat


class ContextEmbeddingAgent(Agent):
    """
    Currently the vocab_method input to this class is unused, and the vocabulary is BERT's tokens.
    """

    def __init__(self, total_cards_amount: int, good_cards_amount: int, embedding_method: str = 'bert',
                 vocab: list = None, dist_metric: str = 'l2'):
        super(ContextEmbeddingAgent, self).__init__()
        self.gca = good_cards_amount
        self.tca = total_cards_amount
        self.model, self.tokenizer = self.initiate_LM(embedding_method)
        self.normalize_embeds = dist_metric == 'cosine_sim'

    def embedder(self, word: List[str] or str, context: str):
        if isinstance(word, str):
            word = [word]
        result = self.embed_with_context(word, context)
        if self.normalize_embeds:
            embedding_norms = torch.norm(result, dim=-1, keepdim=True)
            result = result / embedding_norms
        return result

    @staticmethod
    def known_word(word: str):
        return True

    @staticmethod
    def initiate_LM(lm_model: str = 'bert'):
        """
        fetch the model and tokenizer
        """
        global pretrained_LM
        if not pretrained_LM:
            if lm_model == 'bert':
                model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
                pretrained_LM = (model, tokenizer)
            elif lm_model == 'deberta':
                model = DebertaForMaskedLM.from_pretrained('microsoft/deberta-base', output_hidden_states=True)
                tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', output_hidden_states=True)
                pretrained_LM = (model, tokenizer)
            else:
                raise ValueError(f"lm_model {lm_model} is not an option, choose from \'bert\' or \'deberta\'")
        return pretrained_LM

    @staticmethod
    def initiate_fewshot():
        """
        fetch the model and tokenizer
        """
        global pretrained_fewshot
        if not pretrained_fewshot:
            model = GPTNeoForCausalLM.from_pretrained(GPT_NEO_NAME)
            tokenizer = GPT2Tokenizer.from_pretrained(GPT_NEO_NAME)
            pretrained_fewshot = (model, tokenizer)
        return pretrained_fewshot

    def embed_with_context(self, word_list: List[str], context: str):
        """
        Embed each word using the context. The sentence inputted to the model is "context word".
        Parameters
        ----------
        word_list
            list of strings to embed
        context
            string used as context

        Returns
        -------
        embedding matrix (len_words, embed_dim)
        """
        clue_tokenization = len(self.tokenizer.tokenize(context))
        texts = [f"{context} {word}" for word in word_list]
        model_input = self.tokenizer(texts, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**model_input)
            embeddings = outputs['hidden_states'][-1][:, 1 + clue_tokenization, :]
        return embeddings


class EmbeddingClusterSender(EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str,
                 reduction_method: str = 'centroid'):
        super(EmbeddingClusterSender, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)
        assert reduction_method in ['centroid', 'few-shot-api', 'few-shot-1_token', 'few-shot-greedy', 'fusion']
        cluster_range = min(10, good_cards)
        multiplicity = 15
        # self.cluster_amounts = sum([[self.tca // (i + 1)] * multiplicity for i in range(cluster_range)], start=[])
        self.cluster_amounts = sum([list(range(1, cluster_range + 1)) for _ in range(multiplicity)], start=[])
        self.good_embeds = None
        self.bad_embeds = None
        self.reduction_method = reduction_method
        self.model, self.tokenizer, self.token_indices, self.tokens = None, None, None, None
        if reduction_method == 'few-shot-1_token':
            self.model, self.tokenizer = ContextEmbeddingAgent.initiate_fewshot()
            tokenized_vocab = self.tokenizer(self.vocab).input_ids
            one_token_filter = [(tokenized[0], w) for tokenized, w in zip(tokenized_vocab, vocab) if
                                len(tokenized) == 1]
            self.token_indices, self.tokens = zip(*one_token_filter)
            self.token_indices = list(self.token_indices)
            print(f"amount of one-token words in the vocabulary: {len(self.tokens)}")
        self.tokenized_vocab, self.longest_tokenization = None, None
        if reduction_method == 'few-shot-greedy':
            self.model, self.tokenizer = ContextEmbeddingAgent.initiate_fewshot()
            self.tokenized_vocab = self.tokenizer(self.vocab).input_ids
            self.longest_tokenization = max(map(len, self.tokenized_vocab))  # largest number of tokens in vocab

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
        delimiter = self.tokenizer.eos_token if self.reduction_method == 'few-shot-greedy' else "###"
        # delimiter = "###"
        delimiter = '\n\n'
        # prompt = f"group: \"cat\", \"dog\", \"mouse\". description: \"animals\"" \
        #          f"{delimiter}" \
        #          f"group: \"pants\", \"shirt\", \"hat\". description: \"cloths\"" \
        #          f"{delimiter}" \
        #          f"group: \"one\", \"two\", \"three\", \"four\". description: \"numbers\"" \
        #          f"{delimiter}" \
        #          f"group: " + ', '.join(["\"" + t + "\"" for t in targets]) + ". description: "
        prompt = delimiter.join([f"Words: {', '.join(words)}\nHint: {clue}" for words, clue in [
            (('cat', 'dog', 'mouse'), 'animals'),
            (('pants', 'shirt', 'hat'), 'cloths'),
            (('one', 'two', 'three', 'four'), 'numbers'),
            (targets, '')
        ]])

        # print("prompt")
        # print(prompt)
        # print('\n')

        if self.reduction_method == 'centroid':
            norm_dif = torch.norm(self.embeddings[:-1] - centroid, dim=1)
            norm_dif[[self.word_index_dict[word] for word in words]] = torch.inf
            index = torch.argmin(norm_dif).item()
            closest_word = self.index_word_dict[index]
            return closest_word
        if self.reduction_method == 'few-shot-api':
            # print("prompt: \n" + '-'*12)
            # print(prompt, "\n \n")
            closest_word = rotem_utils.few_shot_huggingface(prompt, delimiter)
            # print("few shot result:", "\n", closest_word)
            return closest_word
        if self.reduction_method == 'fusion':
            # list_length is a hyperparameter
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
        if self.reduction_method == 'few-shot-1_token':
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=input_ids.shape[-1] + 1
            )
            scores, = gen_tokens.scores
            vocab_scores = scores.squeeze(0)[self.token_indices]
            chosen_index = vocab_scores.argmax()
            return self.tokens[chosen_index]
        if self.reduction_method == 'few-shot-greedy':
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=input_ids.shape[-1] + self.longest_tokenization,
                eos_token_id=self.tokenizer.encode(delimiter)[0],
                pad_token_id=self.tokenizer.encode(delimiter)[0]
            )
            result_token_ids = []
            vocab_tokenization = self.tokenized_vocab[:]
            vocab_tokenization = [x for x, w in zip(vocab_tokenization, self.vocab) if
                                  w not in words]  # remove board words
            for i, scores in enumerate(gen_tokens.scores):
                unfinished_relevant_words = [(x, j) for j, x in enumerate(vocab_tokenization) if len(x) > i]
                if len(unfinished_relevant_words) == 0:
                    # print(f"ended vocab_scores to receiver the word {self.tokenizer.decode(result_token_ids)}")
                    return self.tokenizer.decode(result_token_ids)
                vocab_scores = scores.squeeze(0)[[x[i] for x, _ in unfinished_relevant_words]]
                highest_value, chosen_index = vocab_scores.max(dim=-1)
                _, chosen_index = unfinished_relevant_words[chosen_index]  # get index relative to vocab_tokenization
                # chosen_index = vocab_scores.argmax()
                if result_token_ids in vocab_tokenization and scores[0, self.tokenizer.eos_token_id] > highest_value:
                    print(f"gpt ended the sentence to receiver the word {self.tokenizer.decode(result_token_ids)}")
                    return self.tokenizer.decode(result_token_ids)
                chosen_token_id = vocab_tokenization[chosen_index][i]
                result_token_ids.append(chosen_token_id)
                vocab_tokenization = [t for t in vocab_tokenization if t[i] == chosen_token_id]
            raise ValueError(f"no word chosen after longest_tokenization iterations! \n {unfinished_relevant_words}")
        raise ValueError("illegal reduction method " + self.reduction_method)

    def largest_good_cluster(self):
        gca = self.good_embeds.shape[0]
        largest_cluster_size = 0
        largest_cluster_indices = None
        largest_cluster_centroid = None
        data = torch.cat([self.good_embeds, self.bad_embeds], dim=0)
        for cluster_amount in self.cluster_amounts:
            centroids, labels = kmeans2(data, cluster_amount)
            labels = torch.from_numpy(labels)
            good_cluster_labels = set(labels[:gca].tolist()) - set(labels[gca:].tolist())
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
        good_words, bad_words, return_tuple = self.sender_knowledge_forward(words)
        if return_tuple is not None:
            return return_tuple

        self.good_embeds = self.embedder(good_words)
        self.bad_embeds = self.embedder(bad_words)

        centroid, indices, clue_len = self.largest_good_cluster()
        try:
            targets = [good_words[i] for i in indices]
        except IndexError:
            print(1)

        centroid, indices, clue_len = self.largest_good_cluster()
        clue_word = self.reduce_cluster(centroid, indices, targets, good_words + bad_words)

        return clue_word, clue_len, sorted(targets), {}

    def ranked_forward(self, words: tuple, verbose: bool = False):
        good_words, bad_words, return_tuple = self.sender_knowledge_forward(words)
        if return_tuple is not None:
            return return_tuple
        self.good_embeds = self.embedder(good_words)
        self.bad_embeds = self.embedder(bad_words)

        centroid, *_ = self.largest_good_cluster()
        norm_diff = torch.norm(self.embeddings - centroid, dim=1)
        sorted_indices = norm_diff.argsort()
        sorted_vocabulary = [self.vocab[i] for i in sorted_indices]

        return sorted_vocabulary


class ExhaustiveSearchSender(EmbeddingAgent):
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
        good_words, bad_words, return_tuple = self.sender_knowledge_forward(words)
        if return_tuple is not None:
            return return_tuple

        good_embeds = self.embedder(good_words)
        bad_embeds = self.embedder(bad_words)

        # good_diff = torch.norm(self.embeddings[:-1].unsqueeze(1) - good_embeds, dim=-1)  # (N_vocab, num_good_cards)
        # bad_diff = torch.norm(self.embeddings[:-1].unsqueeze(1) - bad_embeds, dim=-1)  # (N_vocab, num_bad_cards)

        good_diff = torch.norm(self.embeddings.unsqueeze(1) - good_embeds, dim=-1)  # (N_vocab, num_good_cards)
        bad_diff = torch.norm(self.embeddings.unsqueeze(1) - bad_embeds, dim=-1)  # (N_vocab, num_bad_cards)

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
        best_word_idx = all_best_indices[best_word_idx]  # that word's index in the entire vocabulary

        clue_word = self.index_word_dict[best_word_idx.item()]

        cluster_words_indices = boolean_closeness_mat[best_word_idx].nonzero(as_tuple=True)[0]
        targets = sorted([good_words[i] for i in cluster_words_indices])
        return clue_word, best_word_cluesize.item(), targets, {'optimal_amount': len(all_best_indices)}


    def ranked_forward(self, words: tuple, verbose: bool = False):
        """
        words = (good_words, bad_words), which are lists of strings.
        """
        good_words, bad_words, return_tuple = self.sender_knowledge_forward(words)
        if return_tuple is not None:
            return return_tuple
        good_embeds = self.embedder(good_words)
        bad_embeds = self.embedder(bad_words)

        # good_diff = torch.norm(self.embeddings[:-1].unsqueeze(1) - good_embeds, dim=-1)  # (N_vocab, num_good_cards)
        # bad_diff = torch.norm(self.embeddings[:-1].unsqueeze(1) - bad_embeds, dim=-1)  # (N_vocab, num_bad_cards)

        good_diff = torch.norm(self.embeddings.unsqueeze(1) - good_embeds, dim=-1)  # (N_vocab, num_good_cards)
        bad_diff = torch.norm(self.embeddings.unsqueeze(1) - bad_embeds, dim=-1)  # (N_vocab, num_bad_cards)

        closest_bad_distance = bad_diff.min(dim=1, keepdim=True)[0]  # distance to the closest bad card - shape (N, 1)
        boolean_closeness_mat = (good_diff < closest_bad_distance)
        close_good_cards = boolean_closeness_mat.sum(dim=1)  # amount of good cards that are closer than the closest
        # bad card - shape (N_vocab)
        close_good_cards[[self.word_index_dict[word] for word in good_words + bad_words]] = 0  # avoid current cards

        # tiebreak - choose the best option among the optimal clues ####################
        all_best_indices = torch.nonzero(close_good_cards == close_good_cards.max()).view(-1)
        best_word_cluesize = close_good_cards[all_best_indices[0]]
        # print("amount of optimal clues:", len(all_best_indices))
        best_good_diff = good_diff[all_best_indices]
        closest_bad_distance = closest_bad_distance[all_best_indices]
        best_good_diff = torch.where(best_good_diff < closest_bad_distance,
                                     best_good_diff,
                                     torch.zeros(best_good_diff.shape, dtype=torch.float))
        scores = self.tie_break_scores(best_good_diff, closest_bad_distance.view(-1))
        all_indices = close_good_cards.sort(descending=True)[1]
        optimal_permutation = scores.sort()[1]
        all_indices[:len(all_best_indices)] = all_indices[:len(all_best_indices)][optimal_permutation]
        sorted_vocabulary = [self.vocab[i] for i in all_indices]

        return sorted_vocabulary


class CompletelyRandomSender(EmbeddingAgent):
    """
    A sender agent which operates as follows - given good and bad cards, finds the set of all words in the vocabulary
    with maximal amount of good cards closer than all bad cards. Out of this set, the agent chosses the word which is
    closest to it's matching good cards.
    """

    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str):
        super(CompletelyRandomSender, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)

    def forward(self, words: tuple, verbose: bool = False):
        """
        words = (good_words, bad_words), which are lists of strings.
        """
        good_words, bad_words, return_tuple = self.sender_knowledge_forward(words)
        if return_tuple is not None:
            return return_tuple
        sample = random.sample(self.vocab, len(good_words) + len(bad_words) + 1)
        clue_word = None
        for word in sample:
            if word not in good_words + bad_words:
                clue_word = word
                break
        clue_size = random.sample(range(len(good_words)), 1)[0] + 1
        targets = random.sample(good_words, clue_size)

        return clue_word, clue_size, sorted(targets), {}

    def ranked_forward(self, words: tuple, verbose: bool = False):
        """
        words = (good_words, bad_words), which are lists of strings.
        """
        sorted_vocabulary = random.sample(self.vocab, len(self.vocab))

        return sorted_vocabulary


class EmbeddingNearestReceiver(EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str):
        super(EmbeddingNearestReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)

    def forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        words, clue_word, clue_amount, board_blue_cards = self.receiver_knowledge_forward(words, clue)
        clue_array = self.embedder(clue_word)
        words_array = self.embedder(words)
        norm_dif = torch.norm(words_array - clue_array, dim=1)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return board_blue_cards + [words[i] for i in indices]

    def faiss_forward(self, words: List[str], clues: List[tuple]):
        # quick batch processing of clues using faiss library. board is constant.
        # knowledge usuage is not supported
        clue_words, clue_sizes = zip(*clues)
        max_cluesize = max(clue_sizes)
        clues_array = self.embedder(list(clue_words))
        words_array = self.embedder(words)
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(words_array.numpy())
        _, I = index.search(clues_array.numpy(), max_cluesize)
        ids = [a[:s].tolist() for a, s in zip(I, clue_sizes)]
        return [[words[i] for i in indices] for indices in ids]

    def ranked_forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        words, clue_word, clue_amount, board_blue_cards = self.receiver_knowledge_forward(words, clue)
        clue_array = self.embedder(clue_word)
        words_array = self.embedder(words)
        norm_diff = torch.norm(words_array - clue_array, dim=1)
        sorted_indices = norm_diff.argsort()
        sorted_vocabulary = [words[i] for i in sorted_indices]
        return board_blue_cards + sorted_vocabulary


class CompletelyRandomReceiver(EmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str):
        super(CompletelyRandomReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)

    def forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        words, clue_word, clue_amount, board_blue_cards = self.receiver_knowledge_forward(words, clue)
        choice = random.sample(words, clue_amount)
        return board_blue_cards + choice

    def ranked_forward(self, words: List[str], clue: tuple):
        """
        words is a shuffled list of all words, clue is a tuple (word, number).
        """
        words, clue_word, clue_amount, board_blue_cards = self.receiver_knowledge_forward(words, clue)
        shuffle = random.sample(words, len(words))
        return board_blue_cards + shuffle


class ContextualizedReceiver(ContextEmbeddingAgent):
    def __init__(self, total_cards: int, good_cards: int, embedding_method: str, vocab: list, dist_metric: str):
        super(ContextualizedReceiver, self).__init__(total_cards, good_cards, embedding_method, vocab, dist_metric)

    def forward(self, words: List[str], clue: tuple):
        words, clue_word, clue_amount, board_blue_cards = self.receiver_knowledge_forward(words, clue)
        clue_array = self.embedder(clue_word, context=clue_word).view(-1)  # (768)
        assert len(clue_array.shape) == 1
        words_array = self.embedder(words, context=clue_word)  # (N_words, 768)
        norm_dif = torch.norm(words_array - clue_array, dim=1)  # (N_word)
        _, indices = torch.topk(norm_dif, clue_amount, largest=False)
        return board_blue_cards + [words[i] for i in indices]
