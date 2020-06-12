# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from os import path
import re
from statistics import mean, stdev, variance

from kaggle_preprocessing import boost_data, read_data
from sklearn.model_selection import KFold
import pandas as pd


# calculate % examples in given data that contains abusive words. returns df
def calc_pct_abusive(data, verbose):
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
    df_columns = ["fold", "oov"]

    # unfortunately all the data is in one folder, so I need to manually pick out the relevant sets here
    sample_types = ["random", "topic", "wordbank"]
    per_sample = 3
    state = 42  # seed for random state

    for s in sample_types:
        for i in range(1, per_sample + 1):
            oov_path = path.join("output/stats/oov", f"oov.{s.lower()}{i}.csv")
            filename = f"train.{s}{i}.csv"

            if path.exists(oov_path):  # check if results file already exists
                print(f"OOV already computed for {filename}. Skipping...")
            else:
                data = read_data(filename, verbose)
                folds = manual_kfold(data, k, state)
                return_list = []
                curr_fold_num = 0

                for f in folds:
                    curr_fold_num += 1
                    curr_fold_name = f"{filename}:fold{curr_fold_num}"
                    train, test = f

                    print(f"Computing metrics for {curr_fold_name}...") if verbose else None
                    train_used, train_unused = get_usage_sets(train, lexicon)
                    test_used, test_unused = get_usage_sets(test, lexicon)

                    # OoV
                    only_in_train = len(train_used & test_used) / len(test_used) * 100  # float
                    oov = 100 - only_in_train  # float

                    # add row to list of rows
                    row = [curr_fold_num, oov]
                    row = [round(x, decimals) for x in row]  # round all per-split metrics used
                    return_list.append(row)

                    # export sets
                    # set_path = path.join("output/stats/oov/sets", f"oov.{s.lower()}{i}.csv")
                    #   return_df.to_csv(oov_path, index=False)  # save results to csv
                    pd.DataFrame(train_used).to_csv(path.join("output/stats/oov/sets",
                                                              f"oov.{s.lower()}{i}.train_used.csv"), index=False)

                    # TODO

                # per-sample stats (on all `k` folds)
                just_nums = [x[1] for x in return_list]  # get only numbers, not names
                avg = ["avg", round(mean(just_nums), decimals)]
                var = ["var", round(variance(just_nums), decimals)]
                std = ["std", round(stdev(just_nums), decimals)]
                return_list.extend([[], avg, var, std])  # separate stats from rest w/ blank row + append stats

                # export per sample
                return_df = pd.DataFrame(return_list, columns=df_columns)  # list -> df
                return_df.to_csv(oov_path, index=False)  # save results to csv
                print(f"OOV metrics computed.\n") if verbose else None


""" OOV Helper Functions """


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


# intersect + complement b/w given df and lexicon
def get_usage_sets(df, lex):
    df_words = get_words_set(df)
    lex = set(lex)

    # words in both df and lexicon (intersect)
    vocab_used = df_words & lex

    # words in lexicon but not df
    vocab_unused = lex - df_words

    return vocab_used, vocab_unused


# given df (train or test), return set of words
def get_words_set(df):
    comments = df["comment_text"].tolist()
    all_words = set()  # init
    regex = re.compile("[^A-Za-z0-9]+", re.IGNORECASE)

    # get all words in test set
    for x in comments:
        comment_words = set(x.split())

        for w in comment_words:
            w_re = re.sub(regex, '', w)  # filter special characters to reduce set size
            all_words.add(w_re.lower())

    return all_words


if __name__ == '__main__':
    decimals = 2  # global variable for output

    calc_oov(k=5, verbose=True)
