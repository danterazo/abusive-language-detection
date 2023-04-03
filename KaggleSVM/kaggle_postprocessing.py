# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
import os
import re
from os import path
from statistics import mean, stdev, variance

import pandas as pd
from sklearn.model_selection import KFold

from kaggle_preprocessing import boost_data


# calculate % examples in given data that contains abusive words. returns df
def calc_pct_abusive(data, decimals, verbose):
    only_abusive = data[data["class"] == 1]  # filter data to only abusive examples

    results_df = pd.DataFrame(columns=["pct_explicit_abusiveonly", "pct_implicit_abusiveonly",
                                       "pct_explicit_alldata", "pct_implicit_alldata", "source_lexicon"])

    lexicon_paths = ["data/lexicon_manual/lexicon.manual.all.explicit.CSV",
                     "data/lexicon_wiegand/lexicon.wiegand.base.explicit.CSV",
                     "data/lexicon_wiegand/lexicon.wiegand.expanded.explicit.CSV"]

    for filename in lexicon_paths:
        print(f"Computing % abusive for {filename}...") if verbose else None
        filename_split = filename.split(".", 3)  # split filename into three parts: path, filename,
        source_name = filename_split[1] + "." + filename_split[2]  # get source name from filename

        explicit_list = open(filename).read().splitlines()  # list of explicitly abusive words
        explicit_abusiveonly = boost_data(only_abusive, data_name="", manual_boost=explicit_list, verbose=False)
        explicit_wholedata = boost_data(data, data_name="", manual_boost=explicit_list, verbose=False)

        pct_explicit_abusiveonly = round(len(explicit_abusiveonly) / len(only_abusive) * 100, decimals)
        pct_implicit_abusiveonly = round(100 - pct_explicit_abusiveonly, decimals)

        pct_explicit_alldata = round(len(explicit_wholedata) / len(data) * 100, decimals)
        pct_implicit_alldata = round(100 - pct_explicit_alldata, decimals)

        results_df.loc[len(results_df)] = [pct_explicit_abusiveonly, pct_implicit_abusiveonly, pct_explicit_alldata, pct_implicit_alldata,
                                           source_name]
        print(f"Computed.\n") if verbose else None

    return results_df


# return percent of words that occur in `test` but NOT `train` splits; oov: out-of-vocabulary
def calc_oov(k, decimals, verbose):
    lexicon = open("data/lexicon_manual/lexicon.manual.all.explicit.csv").read().splitlines()  # read as list
    df_columns = ["fold", "oov.all", "oov.abusive", "oov.non"]
    return_df = pd.DataFrame(columns=df_columns)
    return_df["fold"] = [1, 2, 3, 4, 5, "", "avg", "std", "var"]

    # unfortunately all the data is in one folder, so I need to manually pick out the relevant sets here
    sample_types = ["random", "topic", "wordbank"]
    per_sample = 3
    state = 42  # seed for random state

    sources = ["kaggle", "kaggle_toxic"]
    subsets = ["all", "abusive", "non"]

    for source in sources:
        if source == "kaggle_toxic":
            sample_types = ["random", "wordbank"]

        for s in sample_types:
            for i in range(1, per_sample + 1):
                for subset in subsets:
                    oov_folder = f"output/stats/oov/"
                    filename = f"train.{s}{i}.CSV"

                    if source == "kaggle_toxic":
                        oov_folder = f"output_toxic/stats/oov/"
                        filename = f"train.kaggle_toxic.{s}{i}.CSV"

                    oov_path = path.join(oov_folder, f"oov.{source}.{s.lower()}{i}.CSV")
                    """
                    if path.exists(oov_path):  # check if results file already exists
                        print(f"OOV already computed for {filename}. Skipping...")
                    else:
                    """
                    if source == "kaggle":
                        data = pd.read_csv(f"data/{filename}", names=["class", "comment_text"])
                    else:
                        data = pd.read_csv(f"data/{filename}", header=0)  # toxic dataset

                    """
                    if subset == "non":
                        data = data[data["class"] == 0]  # non-abusive data only
                        print(f"non: {data}")
                    elif subset == "abusive":
                        data = data[data["class"] == 1]  # abusive data only
                        print(f"ab: {data}")
                    """

                    folds = manual_kfold(data, k, state, source, f"{s}{i}")
                    return_list = []
                    curr_fold_num = 0

                    for f in folds:
                        curr_fold_num += 1
                        curr_fold_name = f"{filename}:fold{curr_fold_num}"
                        train, test = f

                        if subset == "non":
                            train = train[train["class"] == 0]  # non-abusive data only
                            test = test[test["class"] == 0]
                            # print(f"non train:\n{train}\nnon test:\n{test}")
                        elif subset == "abusive":
                            train = train[train["class"] == 1]  # abusive data only
                            test = test[test["class"] == 1]
                            # print(f"ab train:\n{train}\nab test:\n{test}")

                        print(f"Computing OOV ({subset}) for {curr_fold_name}...") if verbose else None
                        train_used, train_unused = get_usage_sets(train, lexicon)
                        test_used, test_unused = get_usage_sets(test, lexicon)

                        """
                        train_used = set([1])
                        train_unused = set([1])
                        test_used = set([1])
                        test_unused = set([1])
                        """

                        if len(train[train['class'] == 0]) + len(train[train['class'] == 1]) != len(train) or len(
                                test[test['class'] == 0]) + len(test[test['class'] == 1]) != len(test):
                            print(f"LENGTH CHECK FAILED")

                        # percentage OOV
                        oov_words = set(test_used) - set(train_used)  # used words that appear in test but not in train
                        oov = (len(oov_words) / len(test_used)) * 100  # percentage of words in `test` that don't appear in train

                        # oov = len((set(train_used)& set(test_used)) - set(train_used)) / len(test_used) * 100  # float
                        print(oov)
                        # oov = 100 - only_in_train  # float

                        # add row to list of rows
                        # row = [curr_fold_num, oov]
                        # row = [round(x, decimals) for x in row]  # round all per-split metrics used
                        return_list.append(oov)

                        # export used/unused sets
                        train_used_filename = f"oov.{source}.{subset}.{s.lower()}{i}.fold{curr_fold_num}.train_used.CSV"
                        train_used_dir = f"{oov_folder}/{subset}/train/used"
                        train_used_path = path.join(train_used_dir, train_used_filename)
                        os.makedirs(train_used_dir) if not path.exists(train_used_dir) else None
                        pd.DataFrame(train_used).to_csv(train_used_path, index=False, header=False)

                        train_unused_filename = f"oov.{source}.{subset}.{s.lower()}{i}.fold{curr_fold_num}.train_unused.CSV"
                        train_unused_dir = f"{oov_folder}/{subset}/train/unused"
                        train_unused_path = path.join(train_unused_dir, train_unused_filename)
                        os.makedirs(train_unused_dir) if not path.exists(train_unused_dir) else None
                        pd.DataFrame(train_unused).to_csv(train_unused_path, index=False, header=False)

                        test_used_filename = f"oov.{source}.{subset}.{s.lower()}{i}.fold{curr_fold_num}.test_used.CSV"
                        test_used_dir = f"{oov_folder}/{subset}/test/used"
                        test_used_path = path.join(test_used_dir, test_used_filename)
                        os.makedirs(test_used_dir) if not path.exists(test_used_dir) else None
                        pd.DataFrame(test_used).to_csv(test_used_path, index=False, header=False)

                        test_unused_filename = f"oov.{source}.{subset}.{s.lower()}{i}.fold{curr_fold_num}.test_unused.CSV"
                        test_unused_dir = f"{oov_folder}/{subset}/test/unused"
                        test_unused_path = path.join(test_unused_dir, test_unused_filename)
                        os.makedirs(test_unused_dir) if not path.exists(test_unused_dir) else None
                        pd.DataFrame(test_unused).to_csv(test_unused_path, index=False, header=False)

                    # per-sample stats (on all `k` folds)
                    just_nums = return_list[1:]  # get only numbers, not names
                    avg = round(mean(just_nums), decimals)
                    var = round(variance(just_nums), decimals)
                    std = round(stdev(just_nums), decimals)
                    return_list.extend(["", avg, std, var])  # separate stats from rest w/ blank row + append stats

                    # export per sample
                    return_df[f"oov.{subset}"] = return_list  # append list as row

                    return_df.to_csv(oov_path, index=False)  # save results to csv
                    print(f"OOV metrics computed.\n") if verbose else None


""" OOV Helper Functions """


# manually create test/train splits
def manual_kfold(data, k, state, source, sample_type):
    to_return = []  # array of arrays; [[train, test], [train, test],...]
    splits_index = []

    kf = KFold(n_splits=k, random_state=state, shuffle=True)
    for train_index, test_index in kf.split(data):
        splits_index.append((train_index, test_index))

    for i in range(k):
        train = data.iloc[splits_index[i][0]]
        test = data.iloc[splits_index[i][1]]
        to_return.append((train, test))

        folder = "output"
        if source == "kaggle_toxic":
            folder = "output_toxic"

        train.to_csv(f"{folder}/stats/oov/folds/{sample_type}/oov.{sample_type}.fold{i + 1}.train.CSV", index=False)
        test.to_csv(f"{folder}/stats/oov/folds/{sample_type}/oov.{sample_type}.fold{i + 1}.test.CSV", index=False)

    return to_return


# intersect + complement b/w given df and lexicon
def get_usage_sets(df, lex):
    df_words = get_words_list(df)

    # words in both df and lexicon (intersect)
    # vocab_used = df_words & lex # set, i.e. only unique words
    vocab_used = []
    for word in df_words:
        if word in lex:
            vocab_used.append(word)

    # words in lexicon but not df
    vocab_unused = set(lex) - set(df_words)
    # vocab_unused = [x for x in lex if x not in df_words]

    return vocab_used, vocab_unused


# given df (train or test), return set of words
def get_words_list(df):
    comments = df["comment_text"].tolist()
    all_words = []  # init
    regex = re.compile("[^A-Za-z0-9]+", re.IGNORECASE)

    # get all words in df
    for x in comments:
        comment_words = x.split()

        for w in comment_words:
            w_re = re.sub(regex, '', w)  # filter special characters to reduce set size
            all_words.append(w_re.lower())

    return all_words


if __name__ == '__main__':
    decimals = 2  # round output

    calc_oov(k=5, decimals=decimals, verbose=True)
