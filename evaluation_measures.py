import random
import os
from itertools import product
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import defaultdict

from rotem_main import CodenamesOptions

# constant global variables (no function changes these)
PAPER_RSEULTS_FOLDER = 'paper-dataset_results'
PROLIFIC_RESULTS_FOLDER = 'prolific-dataset_results'


def plot_given_board(good_words: list, bad_words: list, grey_words: list, word_order: list = None, ax=None,
                     show: bool = True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=150)
    # plt.figure(figsize=(6, 2), dpi=150)
    assert word_order is None or set(good_words + bad_words + grey_words) == set(word_order),\
        "word order isn't union of words"
    if word_order is not None:
        words = word_order
    else:
        words = good_words + bad_words + grey_words
        words = random.sample(words, len(words))
    board_words = np.array(words).reshape((2, len(words) // 2))
    board_colors = np.where(np.isin(board_words, good_words), "cyan", "lightpink")
    board_colors[np.isin(board_words, grey_words)] = 'grey'

    table = ax.table(cellText=board_words, cellColours=board_colors, cellLoc='center', loc='center', alpha=0.1)
    table.scale(1, 5)
    table.set_fontsize(12)
    plt.axis('off')

    if show:
        plt.show()


def plot_comparison(chosen_words: list, targets: list, gt_good_words: list, word_order: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=150)
    ax1.axis("off")
    ax2.axis("off")
    plot_given_board(good_words=chosen_words, bad_words=[],
                     grey_words=[w for w in word_order if w not in chosen_words],
                     word_order=word_order, ax=ax1, show=False)
    plot_given_board(good_words=targets, bad_words=[w for w in word_order if w not in gt_good_words],
                     grey_words=[w for w in word_order if w in gt_good_words and w not in targets],
                     word_order=word_order, ax=ax2, show=False)
    ax1.set_title("your choice")
    ax2.set_title("ground truth")
    plt.show()
    plt.waitforbuttonpress()


def receiver_API(words: list, clue: str, num_choices: int, targets: list, gt_good_words: list):
    chosen_words = []
    choice_counter = [0]
    words = random.sample(words, len(words))

    def choose(word, _event):
        if word in chosen_words:
            print(f"The word {word} was already chosen")
            return
        chosen_words.append(word)
        choice_counter[0] += 1
        print(f"You chose the word {word}, choose {num_choices - choice_counter[0]} more")
        if choice_counter[0] == num_choices:
            print("finished! close the plot tab to continue")
            plot_comparison(chosen_words, targets, gt_good_words, words)

    plt.figure(figsize=(6, 2), dpi=150)
    plt.gcf().suptitle(f"Clue word: {clue} (for {num_choices} {'words' if num_choices > 1 else 'word'})")
    board_words = np.array(words).reshape((2, len(words) // 2))
    board_colors = np.copy(board_words).astype('<U16')

    board_words[...] = ''
    board_colors[...] = 'grey'

    table = plt.table(cellText=board_words, cellColours=board_colors, cellLoc='center', loc='center', alpha=0.1)
    table.scale(1, 5)
    table.set_fontsize(12)
    plt.axis('off')

    vertical_locations = [0.2, 0.6]
    cell_size = (0.2, 0.175)
    horizontal_locations = np.linspace(0, 1, len(words) // 2 + 3)[1:-2]
    if len(words) == 4:
        horizontal_locations = [0.2, 0.6]
    elif len(words) == 6:
        horizontal_locations = [0.15, 0.42, 0.67]
    else:
        raise ValueError(f"function receiverAPI received {len(words)} words but only 4 or 6 are allowed")
    axes = []
    for vertical in vertical_locations:
        for horizontal in horizontal_locations:
            axes.append(plt.axes([horizontal, vertical, *cell_size]))

    button_script = ""
    for i in range(len(words)):
        button_script += f"button{i} = Button(axes[{i}], words[{i}])" + "\n"
        button_script += f"button{i}.on_clicked(lambda x: choose(words[{i}], x))" + "\n"
    button_script += "plt.show()"
    gs = globals().copy()
    gs.update(locals())
    exec(button_script, gs, locals())
    return chosen_words


def combine_paper_data(plot_cluesizes: bool = False, verbose: bool = False):
    # simple method to concatenate the data from different M-turk sessions generated in (cite), and some preprocessing
    data_folder = 'CodenamesAMTData'
    dataset_paths = os.listdir(data_folder)
    dfs = []
    for dataset_path in dataset_paths:
        df = pd.read_csv(os.path.join(data_folder, dataset_path))
        columns = [col for col in df.columns if col.startswith("Input") or col.startswith("Answer")]
        df = df[columns]
        dfs.append(df)
    data = pd.concat(dfs)
    data['Input.embedding_name'] = data['Input.embedding_name'].map(lambda x: x[:x.find("Trial")])
    initial_models = data['Input.embedding_name'].unique().tolist()
    if verbose:
        print("inital models:")
        for model in initial_models:
            print(model)
    if plot_cluesizes:
        plt.style.use("bmh")
        counts = defaultdict(int)
        for _, x in data.iterrows():
            good_word_candidates = x[22:].tolist()
            good_words = [w for w in good_word_candidates if w != 'noMoreRelatedWords']
            counts[len(good_words)] += 1
        plt.bar(*zip(*counts.items()))
        plt.title("number of targets histogram")
        plt.xlabel("number of targets")
        plt.show()
    return data, initial_models


def dataset_metric_sender(initial_model: str = 'all', verbose: bool = True, data=None, initial_models=None, **opts_kwargs):
    """
    inital_model is the model used to generate the HumanReceiverDataset we wish to test on. Alternatively,
     inital_model can be 'all'.
    This function evaluates the sender generated by opts_kwargs as explained in the report.

    Note that the sender requires knowing the amount of good cards on the board at initialization, which isn't constant
    across the dataset but is bounded from above by 4. Therefore, we create 4 senders from the same class and use
    the valid instance at each sample.
    """
    if data is None or initial_models is None:
        data, initial_models = combine_paper_data(verbose=False, plot_cluesizes=False)
    assert initial_model == 'all' or initial_model in initial_models, f"inital model {initial_model} doesn't exist"
    data = data if initial_model == 'all' else data[data['Input.embedding_name'] == initial_model]

    t0 = time.perf_counter()
    sender2 = CodenamesOptions(tca=20, gca=2, **opts_kwargs).sender
    t1 = time.perf_counter()
    if verbose:
        print(f"created sender 2 in {t1 - t0} sec")
    t0 = t1

    sender3 = CodenamesOptions(tca=20, gca=3, **opts_kwargs).sender
    t1 = time.perf_counter()
    if verbose:
        print(f"created sender 3 in {t1 - t0} sec")
    t0 = t1

    sender4 = CodenamesOptions(tca=20, gca=4, **opts_kwargs).sender
    t1 = time.perf_counter()
    if verbose:
        print(f"created sender 4 in {t1 - t0} sec")

    sender_dict = {2: sender2, 3: sender3, 4: sender4}

    # now we calculate the measure
    loss = 0
    N = 0
    words_OOV = set()
    if verbose:
        iterator = tqdm(data.iterrows(), total=data.shape[0])
    else:
        iterator = data.iterrows()
    for _, x in iterator:
        words = x[2:22].tolist()
        clue_word = x[1]
        continue_flag = False
        for word in [clue_word] + words:
            if word not in sender2.vocab:
                # print(f"word {word} isn't in the sender's vocabulary")
                words_OOV.add(word)
                continue_flag = True
                break
        if continue_flag:
            continue
        N += 1
        good_word_candidates = x[22:].tolist()
        good_words = [w for w in good_word_candidates if w != 'noMoreRelatedWords']
        bad_words = [w for w in words if w not in good_words]
        sender_input = (good_words, bad_words)
        sender = sender_dict[len(good_words)]

        ranks = sender.ranked_forward(sender_input)
        loss += ranks.index(clue_word)
    loss /= data.shape[0]
    if verbose:
        print(f"loss: {loss} was claculated over {N} samples out of {data.shape[0]}")
        print("words out of vocabulary:")
        for w in words_OOV:
            print(w)
    return loss


def dataset_metric_receiver(initial_model: str = 'all', verbose: bool = True, data=None, initial_models=None, **opts_kwargs):
    """
    Evaluation measure for a receiver, using the HumanReceiverDataset.
    The measure is defined as the Mean Average Precision (MAP) where words chosen by humans are the "relevant documents"
    as described in the report.
    """
    if data is None or initial_models is None:
        data, initial_models = combine_paper_data(verbose=False, plot_cluesizes=False)
    assert initial_model == 'all' or initial_model in initial_models, f"inital model {initial_model} doesn't exist"
    data = data if initial_model == 'all' else data[data['Input.embedding_name'] == initial_model]

    t0 = time.perf_counter()
    receiver = CodenamesOptions(tca=20, gca=4, **opts_kwargs).receiver
    t1 = time.perf_counter()
    if verbose:
        print(f"created receiver in {t1 - t0} sec")

    # now we calculate the measure
    map_score = 0
    N = 0
    words_OOV = set()
    if verbose:
        iterator = tqdm(data.iterrows(), total=data.shape[0])
    else:
        iterator = data.iterrows()
    for _, x in iterator:
        words = x[2:22].tolist()
        clue_word = x[1]
        continue_flag = False
        for word in [clue_word] + words:
            if word not in receiver.vocab:
                # print(f"word {word} isn't in the sender's vocabulary")
                words_OOV.add(word)
                continue_flag = True
                break
        if continue_flag:
            continue
        N += 1
        good_word_candidates = x[22:].tolist()
        good_words = [w for w in good_word_candidates if w != 'noMoreRelatedWords']
        clue_size = len(good_words)
        receiver_input = (words, (clue_word, clue_size))

        ranks = receiver.ranked_forward(*receiver_input)
        ranks_of_good_words = [ranks.index(g_w) + 1 for g_w in good_words]
        # calculation of AP starts here
        AP_score = 0
        for i, good_word_rank in enumerate(sorted(ranks_of_good_words)):
            AP_score += (i + 1) / good_word_rank
        AP_score /= clue_size
        map_score += AP_score  # MAP is the average of AP_scores
    map_score /= N
    if verbose:
        print(f"MAP score: {map_score} was claculated over {N} samples out of {data.shape[0]}")
        print("words out of vocabulary:")
        for w in words_OOV:
            print(w)
    return map_score


def calculate_results(sender_save_path: str, receiver_save_path: str, start_at_batch: int = 0):
    if not os.path.isdir(PAPER_RSEULTS_FOLDER):
        os.mkdir(PAPER_RSEULTS_FOLDER)
    data, initial_models = combine_paper_data(plot_cluesizes=False, verbose=False)
    initial_models = initial_models[:6]
    initial_models.append('all')
    embeddings = ['GloVe', 'word2vec']
    sender_types = ['cluster', 'exhaustive-avg_blue_dist', 'exhaustive-max_blue_dist',
                    'exhaustive-max_radius', 'exhaustive-red_blue_diff', 'random']
    receiver_types = ['embedding', 'random']
    # sender results
    total = len(initial_models) * len(embeddings) * len(sender_types)
    print("total rows for sender results:", total)
    save_every = 10
    all_combinations = list(product(initial_models, sender_types, embeddings))
    batches = [(save_every * i, save_every * (i + 1)) for i in
               range(total // save_every + int(total % save_every != 0))]
    batches = [(i + 1, *x) for i, x in enumerate(batches)]
    for i, batch_start, batch_end in batches[start_at_batch:]:
        print("beginning to calculate batch", i)
        initial_model_list = []
        sender_type_list = []
        sender_embedding_list = []
        loss_list = []
        for initial_model, sender_type, sender_embed in tqdm(all_combinations[batch_start:batch_end], total=save_every):
            if sender_type in ['cluster', 'random']:
                sender_class = sender_type
                kwargs = {}
            else:
                sender_class, tiebreak = sender_type.split("-")
                kwargs = {'exhaustive_tiebreak': tiebreak}
            loss = dataset_metric_sender(initial_model=initial_model, verbose=False, data=data,
                                         initial_models=initial_models, sender_type=sender_class,
                                         agent_vocab='GloVe', sender_emb_method=sender_embed,
                                         receiver_emb_method='word2vec' if sender_embed == 'GloVe' else 'GloVe',
                                         **kwargs)
            initial_model_list.append(initial_model)
            sender_type_list.append(sender_type)
            sender_embedding_list.append(sender_embed)
            loss_list.append(loss)
        sender_results = pd.DataFrame({'initial_model': initial_model_list, 'sender_type': sender_type_list,
                                       'sender_embedding': sender_embedding_list, 'loss': loss_list})
        # sender_results.to_pickle(os.path.join("results", f"{sender_save_path}_batch_{i}"))
        sender_results.to_csv(os.path.join(PAPER_RSEULTS_FOLDER, f"{sender_save_path}_batch_{i}") + ".csv")
        print(f"saved shape {sender_results.shape} to file {sender_save_path}_batch_{i}")

    # receiver results
    total = len(initial_models) * len(embeddings) * len(receiver_types)
    print("total rows for sender results:", total)
    save_every = 10
    all_combinations = list(product(initial_models, receiver_types, embeddings))
    batches = [(save_every * i, save_every * (i + 1)) for i in
               range(total // save_every + int(total % save_every != 0))]
    batches = [(i + 1, *x) for i, x in enumerate(batches)]
    for i, batch_start, batch_end in batches[start_at_batch:]:
        print("beginning to calculate batch", i)
        initial_model_list = []
        receiver_type_list = []
        receiver_embedding_list = []
        score_list = []

        for initial_model, receiver_type, receiver_embed in tqdm(all_combinations[batch_start:batch_end], total=save_every):
            score = dataset_metric_receiver(initial_model=initial_model, verbose=False, data=data,
                                            initial_models=initial_models, receiver_type=receiver_type,
                                            agent_vocab='GloVe', receiver_emb_method=receiver_embed,
                                            sender_emb_method='word2vec' if receiver_embed == 'GloVe' else 'GloVe')
            initial_model_list.append(initial_model)
            receiver_type_list.append(receiver_type)
            receiver_embedding_list.append(receiver_embed)
            score_list.append(score)
        receiver_results = pd.DataFrame({'initial_model': initial_model_list, 'receiver_type': receiver_type_list,
                                         'receiver embedding': receiver_embedding_list, 'score': score_list})
        # receiver_results.to_pickle(os.path.join("results", f"{receiver_save_path}_batch_{i}"))
        receiver_results.to_csv(os.path.join(PAPER_RSEULTS_FOLDER, f"{receiver_save_path}_batch_{i}") + ".csv")
        print(f"saved shape {receiver_results.shape} to file {receiver_save_path}_batch_{i}")


def combine_batches(folder_path: str, saved_path: str, save_csv: bool = False):
    """
    combine the batches created at `calculate_results` into one dataframe.
    """
    all_files = os.listdir(folder_path)
    batches = sorted([file for file in all_files if file.startswith(f"{saved_path}_batch_")])
    dfs = [pd.read_csv(os.path.join(folder_path, file_path)) for file_path in batches]
    df = pd.concat(dfs).reset_index(drop=True)
    if save_csv:
        df.to_csv(os.path.join(folder_path, saved_path.split('.')[0] + ".csv"))
    return df


def analysis_main(saved_path: str, comparison_column: str = 'sender_embedding',
                  constant_column: str = 'sender_type', constant_value: str = 'cluster'):
    """
    create a bar plot with data collected and calculated over the HumanReceiverDataset.
    """
    df: pd.DataFrame
    sender_plot_bool = comparison_column.startswith("sender")
    y_label = 'loss' if sender_plot_bool else 'score'
    df = combine_batches(PAPER_RSEULTS_FOLDER, saved_path)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.head(5))

    x_column = 'initial_model'
    r_df = df[df[constant_column] == constant_value][[x_column, comparison_column, y_label]]
    # # this part is only relevant in debug where we don't have the entire results dataframe ###########
    # for initial_model, comparison_val in product(initial_models, r_df[comparison_column]):
    #     if not ((r_df[x_column] == initial_model) * (r_df[comparison_column] == comparison_val)).any():
    #         r_df = r_df.append({x_column: initial_model, comparison_column: comparison_val, 'loss': 1}, ignore_index=True)
    #         print(f"added row for {initial_model=}, {comparison_column}={comparison_val}")
    # ######################################################################################################

    t_df = r_df.pivot(index=x_column, columns=comparison_column,
                      values=y_label)

    def rename(model_name: str):
        if 'W' not in model_name:
            assert model_name == 'all'
            return model_name
        base_name = model_name[:model_name.index('W')]
        if not "WithoutHeuristics" in model_name:
            base_name += "_H"
        if not "WithoutKimFx" in model_name:
            base_name += "_KimFX"
        return base_name

    print("\n", t_df.index.tolist())
    t_df.index = list(map(rename, t_df.index.tolist()))
    print(t_df.index.tolist())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(t_df.head(5))

    t_df.plot.bar(rot=0, figsize=(15, 8))
    extra_string = f'. {constant_column}={constant_value}' if sender_plot_bool else ''
    plt.title(f"{y_label} comparison of {comparison_column}, for different initial models{extra_string}", fontsize=20)
    plt.ylabel(y_label)
    plt.xlabel("initial model")
    plt.show()


def prolific_metric_sender(verbose: bool = True, data=None, **opts_kwargs):
    """
    The evaluation measure for a sender, using the data we have collected from Prolific.
    Assumes the data has the form created by "prolific_to_df".
    """
    data: pd.DataFrame
    if data is None:
        data = prolific_to_df(verbose)
    sender = CodenamesOptions(tca=6, gca=3, **opts_kwargs).sender
    loss = 0
    N = 0
    iterator = data.groupby(['good_words', 'bad_words'])
    if verbose:
        iterator = tqdm(iterator)
    for (good_words, bad_words), grouped_data in iterator:
        continue_flag = False
        for word in good_words + bad_words:
            if word not in sender.vocab:
                continue_flag = True
                break
        if continue_flag:
            continue
        ranks = sender.ranked_forward((list(good_words), list(bad_words)))
        for i, row in grouped_data.iterrows():
            if row.clue_word not in sender.vocab:
                continue
            N += 1
            loss += ranks.index(row.clue_word)
    assert N != 0, "The vocabulary doesn't include any row from the dataset!"
    loss /= N
    print(f"calculated loss {loss} with {N=} samples out of {data.shape[0]}")
    return loss


def prolific_metric_receiver(verbose: bool = True, data=None, **opts_kwargs):
    """
    The evaluation measure for a receiver, using the data we have collected from Prolific.
    Assumes the data has the form created by "prolific_to_df".
    """
    TARGET_SCORE = 1
    BLUE_SCORE = 0
    RED_SCORE = -1
    data: pd.DataFrame
    if data is None:
        data = prolific_to_df(verbose)
    receiver = CodenamesOptions(tca=6, gca=3, **opts_kwargs).receiver
    map_score = 0
    N = 0
    iterator = tqdm(data.iterrows(), total=data.shape[0]) if verbose else data.iterrows()
    for _, row in iterator:
        good_words, bad_words, clue_word, targets = row
        continue_flag = False
        for word in good_words + bad_words + (clue_word,):
            if word not in receiver.vocab:
                continue_flag = True
                break
        if continue_flag:
            continue
        N += 1
        guesses = receiver(list(good_words) + list(bad_words), (clue_word, len(targets)))
        current_score = 0
        for guess in guesses:
            if guess in targets:
                current_score += TARGET_SCORE
            elif guess in good_words:
                current_score += BLUE_SCORE
            else:
                current_score += RED_SCORE
        current_score /= len(guesses)
        map_score += current_score
    assert N != 0, "The vocabulary doesn't include any row from the dataset!"
    map_score /= N
    print(f"calculated map score {map_score} with {N=} samples out of {data.shape[0]}")
    return map_score


def calculate_results_prolific(sender_save_path: str, receiver_save_path: str, start_at_batch: int = 0):
    if not os.path.isdir(PROLIFIC_RESULTS_FOLDER):
        os.mkdir(PROLIFIC_RESULTS_FOLDER)
    data = prolific_to_df(verbose=False)
    embeddings = ['GloVe', 'word2vec']
    sender_types = ['cluster', 'exhaustive-avg_blue_dist', 'exhaustive-max_blue_dist',
                    'exhaustive-max_radius', 'exhaustive-red_blue_diff', 'random']
    receiver_types = ['embedding', 'random']
    # sender results
    total = len(embeddings) * len(sender_types)
    print("total rows for sender results:", total)
    save_every = 10
    all_combinations = list(product(sender_types, embeddings))
    batches = [(save_every * i, save_every * (i + 1)) for i in
               range(total // save_every + int(total % save_every != 0))]
    batches = [(i + 1, *x) for i, x in enumerate(batches)]
    for i, batch_start, batch_end in batches[start_at_batch:]:
        print("beginning to calculate batch", i)
        sender_type_list = []
        sender_embedding_list = []
        loss_list = []
        for sender_type, sender_embed in tqdm(all_combinations[batch_start:batch_end], total=save_every):
            if sender_type in ['cluster', 'random']:
                sender_class = sender_type
                kwargs = {}
            else:
                sender_class, tiebreak = sender_type.split("-")
                kwargs = {'exhaustive_tiebreak': tiebreak}
            loss = prolific_metric_sender(verbose=False, data=data, sender_type=sender_class,
                                          agent_vocab='GloVe', sender_emb_method=sender_embed,
                                          receiver_emb_method='word2vec' if sender_embed == 'GloVe' else 'GloVe',
                                          **kwargs)
            sender_type_list.append(sender_type)
            sender_embedding_list.append(sender_embed)
            loss_list.append(loss)
        sender_results = pd.DataFrame({'sender_type': sender_type_list, 'sender_embedding': sender_embedding_list,
                                       'loss': loss_list})
        sender_results.to_csv(os.path.join(PROLIFIC_RESULTS_FOLDER, f"{sender_save_path}_batch_{i}.csv"))
        print(f"saved shape {sender_results.shape} to file {sender_save_path}_batch_{i}")

    # receiver results
    total = len(embeddings) * len(receiver_types)
    print("total rows for sender results:", total)
    save_every = 10
    all_combinations = list(product(receiver_types, embeddings))
    batches = [(save_every * i, save_every * (i + 1)) for i in
               range(total // save_every + int(total % save_every != 0))]
    batches = [(i + 1, *x) for i, x in enumerate(batches)]
    for i, batch_start, batch_end in batches[start_at_batch:]:
        print("beginning to calculate batch", i)
        receiver_embedding_list = []
        receiver_type_list = []
        score_list = []

        for receiver_type, receiver_embed in tqdm(all_combinations[batch_start:batch_end], total=save_every):
            score = prolific_metric_receiver(verbose=False, data=data, receiver_type=receiver_type,
                                             agent_vocab='GloVe', receiver_emb_method=receiver_embed,
                                             sender_emb_method='word2vec' if receiver_embed == 'GloVe' else 'GloVe')
            receiver_embedding_list.append(receiver_embed)
            receiver_type_list.append(receiver_type)
            score_list.append(score)
        receiver_results = pd.DataFrame({'receiver embedding': receiver_embedding_list,
                                         'receiver_type': receiver_type_list,
                                         'score': score_list})
        receiver_results.to_csv(os.path.join(PROLIFIC_RESULTS_FOLDER, f"{receiver_save_path}_batch_{i}.csv"))
        print(f"saved shape {receiver_results.shape} to file {receiver_save_path}_batch_{i}")


def prolific_to_df(verbose: bool = True):
    """
    transforms the prolific data into a DataFrame with the columns
    good_word-1, ..., good_word-3, bad_word-1, ..., bad_word-3, clue_word, target-1, ..., target-3
    target-i can also be "NoMoreTargets".
    """
    prolific_folder = 'prolific_data'
    demo_folder = 'prolific_demo_data'
    main_data_folder = 'collected_data'
    word_colors_path = os.path.join(prolific_folder, 'word_colors.csv')

    # extract the dictionaries of good/bad words at each question from word_colors_path
    wc_df = pd.read_csv(word_colors_path)
    wc_df['good_words'] = wc_df[[f"blue{i}" for i in range(1, 4)]].agg(tuple, axis=1)
    wc_df['bad_words'] = wc_df[[f"red{i}" for i in range(1, 4)]].agg(tuple, axis=1)
    wc_df = wc_df[['index', 'good_words', 'bad_words']]
    wc_df['index'] = wc_df['index'].map(lambda key: key.split('_')[0])  # f0q2_d -> f0q2
    wc_df.set_index('index', inplace=True)
    words_dict = wc_df.to_dict()
    good_words_dict = words_dict['good_words']
    bad_words_dict = words_dict['bad_words']
    if verbose:
        print(f"{len(good_words_dict)=}, {len(good_words_dict['f0q1'])=}")

    # find which rows should be rejected from the final dataframe
    demo_full_path = os.path.join(prolific_folder, demo_folder)
    bad_ids = set()
    for demo_file in os.listdir(demo_full_path):
        filepath = os.path.join(demo_full_path, demo_file)
        demo_df = pd.read_csv(filepath)
        to_reject = demo_df['participant_id'][demo_df['status'] == 'REJECTED']
        bad_ids.update(to_reject.tolist())

    # preprocess and concatenate tables at main_data_folder
    data_full_path = os.path.join(prolific_folder, main_data_folder)
    dfs = []
    for filename in os.listdir(data_full_path):
        filepath = os.path.join(data_full_path, filename)
        form_number = filename[4]
        df = pd.read_csv(filepath)
        mask = df.apply(lambda row: row[1] not in bad_ids, axis=1)
        df = df[mask]
        df = df[df.columns[5:]]  # skip initial question and first exmaple board
        board_dfs = []
        for i in range(0, len(df.columns), 3):
            board_df = df[df.columns[i:i + 3]].rename(lambda name: name.split('.')[0], axis='columns')
            key = f"f{form_number}q{i // 3}"
            clue_word_column, targets_word_column = board_df.columns[:2]
            board_df['good_words'] = [good_words_dict[key]] * board_df.shape[0]
            board_df['bad_words'] = [bad_words_dict[key]] * board_df.shape[0]
            board_df['targets'] = board_df[targets_word_column].map(lambda targets: targets.split(';'))
            board_df['clue_word'] = board_df[clue_word_column].str.lower()
            board_df = board_df[['good_words', 'bad_words', 'clue_word', 'targets']]
            board_dfs.append(board_df)
        concatenated_df = pd.concat(board_dfs)
        dfs.append(concatenated_df)
    data = pd.concat(dfs)
    if verbose:
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(data[25:35])

    return data


def analysis_prolific(saved_path: str, x_column: str = 'sender_type'):
    df = combine_batches(PROLIFIC_RESULTS_FOLDER, saved_path)
    comparison_column = {'sender_type': 'sender_embedding', 'sender_embedding': 'sender_type',
                         'receiver embedding': 'receiver_type', 'receiver_type': 'receiver embedding'}[x_column]
    y_label = 'loss' if x_column.startswith("sender") else 'score'
    p_df = df.pivot(index=x_column, columns=comparison_column, values=y_label)
    title = f"{y_label} of {comparison_column} against {x_column}, measured by our dataset"
    p_df.plot.bar(rot=0, figsize=(15, 8))
    plt.title(title, fontsize=20)
    plt.ylabel(y_label)
    plt.show()


def evaluate_agent(role: str, verbose: bool = True, **opts_kwargs):
    assert role in ['sender', 'receiver'], f"role should be \'sender\' or \'receiver\', not allowed: {role}"

    prolific_func = {'sender': prolific_metric_sender, 'receiver': prolific_metric_receiver}[role]
    prolific_value = prolific_func(verbose=verbose, **opts_kwargs)

    paper_func = {'sender': dataset_metric_sender, 'receiver': dataset_metric_receiver}[role]
    paper_value = paper_func('all', verbose=verbose, **opts_kwargs)
    return {'HumanSenderDataset': prolific_value, 'HumanReceiverDataset': paper_value}
