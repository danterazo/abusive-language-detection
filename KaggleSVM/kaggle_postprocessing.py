# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from os import path
import os
import re
from statistics import mean, stdev, variance

from kaggle_preprocessing import boost_data, read_data
from sklearn.model_selection import KFold
import pandas as pd


# calculate % examples in given data that contains abusive words. returns df
def calc_pct_abusive(data, decimals, verbose):
    data_abusive = data[data["class"] == 1]  # filter data to only abusive examples

    results_df = pd.DataFrame(columns=["pct_explicit", "pct_implicit", "source_lexicon"])

    lexicon_paths = ["data/lexicon_manual/lexicon.manual.all.explicit.CSV",
                     "data/lexicon_wiegand/lexicon.wiegand.base.explicit.CSV",
                     "data/lexicon_wiegand/lexicon.wiegand.expanded.explicit.CSV"]

    for filename in lexicon_paths:
        print(f"Computing % abusive for {filename}...") if verbose else None
        filename_split = filename.split(".", 3)  # split filename into three parts: path, filename,
        source_name = filename_split[1] + "." + filename_split[2]  # get source name from filename

        explicit_list = open(filename).read().splitlines()  # list of explicitly abusive words
        data_explicit = boost_data(data_abusive, "", False, manual_boost=explicit_list)

        pct_explicit = round(len(data_explicit) / len(data_abusive) * 100, decimals)
        pct_implicit = round(100 - pct_explicit, decimals)

        results_df.loc[len(results_df)] = [pct_explicit, pct_implicit, source_name]
        print(f"Computed.\n") if verbose else None

    return results_df

# return percent of words that occur in `test` but NOT `train` splits
# oov: out-of-vocabulary
def calc_oov(k, decimals, verbose):
    lexicon = open("data/lexicon_manual/lexicon.manual.all.explicit.csv").read().splitlines()  # read as list
    df_columns = ["fold", "oov"]

    # unfortunately all the data is in one folder, so I need to manually pick out the relevant sets here
    sample_types = ["random", "topic", "wordbank"]
    per_sample = 3
    state = 42  # seed for random state

    for s in sample_types:
        for i in range(1, per_sample + 1):
            oov_folder = "output/stats/oov"
            oov_path = path.join(oov_folder, f"oov.{s.lower()}{i}.CSV")
            filename = f"train.{s}{i}.CSV"

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

                    # export used/unused sets
                    train_used_filename = f"oov.{s.lower()}{i}.fold{curr_fold_num}.train_used.CSV"
                    train_used_dir = f"{oov_folder}/train/used"
                    train_used_path = path.join(train_used_dir, train_used_filename)
                    os.makedirs(train_used_dir) if not path.exists(train_used_dir) else None
                    pd.DataFrame(train_used).to_csv(train_used_path, index=False, header=False)

                    train_unused_filename = f"oov.{s.lower()}{i}.fold{curr_fold_num}.train_unused.CSV"
                    train_unused_dir = f"{oov_folder}/train/unused"
                    train_unused_path = path.join(train_unused_dir, train_unused_filename)
                    os.makedirs(train_unused_dir) if not path.exists(train_unused_dir) else None
                    pd.DataFrame(train_unused).to_csv(train_unused_path, index=False, header=False)

                    test_used_filename = f"oov.{s.lower()}{i}.fold{curr_fold_num}.test_used.CSV"
                    test_used_dir = f"{oov_folder}/test/used"
                    test_used_path = path.join(test_used_dir, test_used_filename)
                    os.makedirs(test_used_dir) if not path.exists(test_used_dir) else None
                    pd.DataFrame(test_used).to_csv(test_used_path, index=False, header=False)

                    test_unused_filename = f"oov.{s.lower()}{i}.fold{curr_fold_num}.test_unused.CSV"
                    test_unused_dir = f"{oov_folder}/test/unused"
                    test_unused_path = path.join(test_unused_dir, test_unused_filename)
                    os.makedirs(test_unused_dir) if not path.exists(test_unused_dir) else None
                    pd.DataFrame(test_unused).to_csv(test_unused_path, index=False, header=False)

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
    decimals = 2  # round output

    calc_oov(k=5, decimals=decimals, verbose=True)
