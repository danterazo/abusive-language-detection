# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_preprocessing import boost_data, read_data
from sklearn.model_selection import KFold
import pandas as pd


# calculate % examples in given data that contains abusive words. returns df
# TODO: multiprocessing library
def percent_abusive(data):
    results_df = pd.DataFrame(columns=["pct_abusive", "source_lexicon"])

    lexicon_filenames = ["data/lexicon_manual/lexicon.manual.all.abusive.csv",
                         "data/lexicon_wiegand/lexicon.wiegand.base.abusive.csv",
                         "data/lexicon_wiegand/lexicon.wiegand.expanded.abusive.csv"]

    for f in lexicon_filenames:
        f_split = f.split(".", 3)
        source = f_split[1] + "." + f_split[2]  # get source name from filename

        boost_list = open(f).read().splitlines()  # read file as list
        boosted_df = boost_data(data, "", False, manual_boost=boost_list)
        pct = round(len(boosted_df) / len(data) * 100, 2)

        results_df.loc[len(results_df)] = [pct, source]

    return results_df


# return percent of words that occur in `test` but NOT `train` splits
# oov: out-of-vocabulary
def calc_oov(k):
    manual_lexicon = open("data/lexicon_manual/lexicon.manual.all.abusive.csv").read().splitlines()  # read as list

    # unfortunately all the data is in one folder, so I need to manually pick out the relevant sets here
    per_sample = 3
    sample_types = ["random", "topic", "wordbank"]

    for s in sample_types:
        for i in range(1, per_sample + 1):
            filename = f"train.{s}{i}.csv"
            data = read_data(filename, verbose=False)
            folds = manual_kfold(data, k)

            for f in folds:
                train = f[0]
                test = f[1]

                # set operations
                vocab_used, vocab_unused, oov = set_ops(test, manual_lexicon)

                # filter on vocab_used
                test_filtered = boost_data(train, "", False, manual_boost=vocab_used)

                # TODO: calc percent + return
                # TODO: export

                print(test_filtered)

    # calculate results

    pass


# manually create test/train splits
def manual_kfold(data, k):
    to_return = []  # array of arrays; [[train, test], [train, test],...]
    splits_index = []

    kf = KFold(n_splits=k, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(data):
        splits_index.append([train_index, test_index])

    for i in range(k):
        train = data.iloc[splits_index[i][0]]
        test = data.iloc[splits_index[i][1]]
        to_return.append([train, test])

    return to_return


# given df (train or test), return list of words
def get_words_list(data):
    comments = data["comment_text"].tolist()
    words = []

    # get all words in test set
    for x in comments:
        words = x.split()

        for w in words:
            words.append(w)

    words = list(set(words))  # remove dupes
    return words


# intersect + 2x complements b/w test words and lexicon
def set_ops(test, lex):
    lex = set(lex)

    test_words = get_words_list(test)

    # words in both test and lexicon (intersect)
    vocab_used = set(test_words) & set(lex)

    # words in lexicon but not test (not OOV)
    vocab_unused = lex.difference(vocab_used)

    # words in test but not lexicon (out-of-vocabulary, OOV)
    oov = vocab_used.difference(lex)

    return vocab_used, vocab_unused, oov


if __name__ == '__main__':
    print(calc_oov(k=5))
