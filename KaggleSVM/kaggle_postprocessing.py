# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_preprocessing import boost_data, read_data
from kaggle_build import export_df
from sklearn.model_selection import KFold
import pandas as pd
from os import path
import os.path
import re

decimals = 2  # global variable for output


# calculate % examples in given data that contains abusive words. returns df
# TODO: multiprocessing library
def percent_abusive(data, verbose):
    results_df = pd.DataFrame(columns=["pct_abusive", "source_lexicon"])

    lexicon_filenames = ["data/lexicon_manual/lexicon.manual.all.abusive.csv",
                         "data/lexicon_wiegand/lexicon.wiegand.base.abusive.csv",
                         "data/lexicon_wiegand/lexicon.wiegand.expanded.abusive.csv"]

    # NOTE: tried multithreading, but performance improvement was negligible (~10s) and it wasn't reliable
    for f in lexicon_filenames:
        print(f"Computing % abusive for {f}...") if verbose else None
        f_split = f.split(".", 3)
        source = f_split[1] + "." + f_split[2]  # get source name from filename

        boost_list = open(f).read().splitlines()  # read file as list
        boosted_df = boost_data(data, "", False, manual_boost=boost_list)
        pct = round(len(boosted_df) / len(data) * 100, decimals)

        results_df.loc[len(results_df)] = [pct, source]
        print(f"Computed.\n") if verbose else None

    return results_df


# return percent of words that occur in `test` but NOT `train` splits
# oov: out-of-vocabulary
def calc_oov(k, verbose):
    lexicon = open("data/lexicon_manual/lexicon.manual.all.abusive.csv").read().splitlines()  # read as list
    df_columns = ["fold", "train_pct_used", "train_pct_unused", "train_num_used", "train_num_unused",
                  "test_pct_used", "test_pct_unused", "test_num_used", "test_num_unused", "used_ratio",
                  "unused_ratio", "only_in_train", "only_in_test"]

    # unfortunately all the data is in one folder, so I need to manually pick out the relevant sets here
    sample_types = ["random", "topic", "wordbank"]
    per_sample = 3
    state = 42  # seed for random state

    for s in sample_types:
        for i in range(1, per_sample + 1):
            oov_path = os.path.join("output/stats/oov", f"oov.{s.lower()}{i}.csv")
            filename = f"train.{s}{i}.csv"

            if path.exists(oov_path):  # check if results file already exists
                print(f"OOV already computed for {filename}. Skipping...")  # TODO: are we really calculating OOV?
            else:
                data = read_data(filename, verbose)
                folds = manual_kfold(data, k, state)
                return_list = []
                curr_fold_num = 0

                for f in folds:
                    curr_fold_name = f"{filename}:fold{curr_fold_num}"
                    curr_fold_num += 1
                    train, test = f

                    print(f"Computing metrics for {curr_fold_name}...") if verbose else None
                    train_metrics = lexicon_usage_metrics(train, lexicon, curr_fold_name, len(train))
                    train_pct_used, train_pct_unused, train_num_used, train_num_unused, train_used = train_metrics

                    test_metrics = lexicon_usage_metrics(test, lexicon, curr_fold_name, len(train))
                    test_pct_used, test_pct_unused, test_num_used, test_num_unused, test_used = test_metrics

                    # ratio: # of words in both train and test divided by # of words in test
                    only_in_train = round(len(train_used & test_used) / len(test_used) * 100, decimals)
                    only_in_test = round(100 - only_in_train, decimals)

                    # ratio of used words between train and test. if 1, then equal
                    used_ratio = train_pct_used / test_pct_used
                    unused_ratio = train_pct_unused / test_pct_unused

                    row = [curr_fold_num] + train_metrics[0:4] + test_metrics[0:4] + [used_ratio, unused_ratio,
                                                                                      only_in_train, only_in_test]
                    row = [round(x, decimals) for x in row]
                    return_list.append(row)

                # export per sample
                return_df = pd.DataFrame(return_list, columns=df_columns)
                return_df.to_csv(oov_path, index=False)  # save results to csv
                print(f"OOV metrics computed.\n") if verbose else None


""" Metrics-DF Helper Functions """


# manually create test/train splits
def manual_kfold(data, k, state):
    to_return = []  # array of arrays; [[train, test], [train, test],...]
    splits_index = []

    kf = KFold(n_splits=k, random_state=state, shuffle=True)
    for train_index, test_index in kf.split(data):
        splits_index.append([train_index, test_index])

    for i in range(k):
        train = data.iloc[splits_index[i][0]]
        test = data.iloc[splits_index[i][1]]
        to_return.append([train, test])

    return to_return


# given df (train or test), return set of words
def get_words_set(data):
    comments = data["comment_text"].tolist()
    all_words = set()  # init
    regex = re.compile("[^A-Za-z0-9]+", re.IGNORECASE)

    # get all words in test set
    for x in comments:
        comment_words = set(x.split())

        for w in comment_words:
            w_re = re.sub(regex, '', w)  # filter special characters to reduce set size
            all_words.add(w_re.lower())

    # TODO: verify that this works correctly
    return all_words


# intersect + 2x complements b/w given df words and lexicon
def set_ops(df, lex):
    lex = set(lex)

    df_words = get_words_set(df)

    # words in both test and lexicon (intersect)
    vocab_used = df_words & lex

    # words in lexicon but not test (not OOV)
    vocab_unused = lex - df_words

    # words in test but not lexicon (out-of-vocabulary, OOV)
    oov = df_words - lex
    # oov = vocab_used - vocab_unused
    # print(len(oov))
    # TODO: why is OOV always 80%?

    return vocab_used, vocab_unused, oov


# return %s for main metrics function
def lexicon_usage_metrics(df, lexicon, curr_fold_name, source_len):
    # set operations
    used, unused, train_oov = set_ops(df, lexicon)

    # TODO: multithreading
    # filter on vocab_used
    used = boost_data(df, curr_fold_name, False, manual_boost=used)
    num_used = len(used)
    pct_used = round(len(used) / source_len * 100, decimals)

    # filter on vocab_unused
    unused = boost_data(df, curr_fold_name, False, manual_boost=unused)
    num_unused = len(unused)
    pct_unused = round(len(unused) / source_len * 100, decimals)

    # filter on OOV
    # TODO: debug
    # train_oov = boost_data(train, curr_fold_name, verbose, manual_boost=test_oov)
    # pct_oov = round(len(train_oov) / train_len * 100, decimals)

    # TODO: refactor all this

    used_set = get_words_set(used)
    return [pct_used, pct_unused, num_used, num_unused, used_set]


if __name__ == '__main__':
    calc_oov(k=5, verbose=True)
