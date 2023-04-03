import re
from statistics import mean

import pandas as pd


def get_oov():
    # configuration
    dataset = "kaggle_toxic"  # toggle: "kaggle" (kaggle large) or "kaggle_toxic" (kaggle original)
    sample_type = ["random", "topic", "wordbank"]
    subsets = ["all", "non", "abusive"]
    i = 3  # number of sets per sample type
    k = 5  # k-fold cross validation
    print_avg = 1  # if 1, print sample_type averages instead of per-set score
    # no more configuration beyond this point

    if dataset == "kaggle_toxic":
        sample_type = ["random", "wordbank"]  # remove "topic" if using original kaggle set

    # these for-loops will generate all the results you need
    for sample in sample_type:  # sample type loop (e.g. the `random` in random1)
        print(f"{dataset}.{sample} Sets =======") if print_avg == 1 else None
        avg_all = []  # average for all sets of given sample type
        avg_non = []
        avg_abusive = []

        for set_num in range(1, i + 1):  # set number loop (e.g. the `1` in random1)
            print(f"{dataset}.{sample}{set_num} Set =======") if print_avg == 0 else None

            for subset in subsets:  # subsets loop (e.g. "all", "non-abusive", "abusive")
                list_of_oovs = []  # will hold OOV for each k. then the list will be averaged

                for k in range(1, k + 1):  # fold loop (k is the total number of folds)
                    train = pd.read_csv(f"{dataset}/{sample}{set_num}/oov.{sample}{set_num}.fold{k}.train.CSV")
                    test = pd.read_csv(f"{dataset}/{sample}{set_num}/oov.{sample}{set_num}.fold{k}.test.CSV")

                    # filter data if necessary
                    if subset == "non":
                        train = train[train["class"] == 0]  # get only non-abusive
                        test = test[test["class"] == 0]
                    elif subset == "abusive":
                        train = train[train["class"] == 1]  # get only abusive
                        test = test[test["class"] == 1]

                    # get used words in train
                    regex = re.compile("[^A-Za-z0-9]+", re.IGNORECASE)
                    train_used = []
                    for comment in train["comment_text"].tolist():
                        for word in comment.split():
                            w_re = re.sub(regex, '', word)  # filter special characters

                            if len(w_re) > 0:
                                train_used.append(w_re.lower())

                    # get used words in test
                    test_used = []
                    for comment in test["comment_text"].tolist():
                        for word in comment.split():
                            w_re = re.sub(regex, '', word)  # filter special characters

                            if len(w_re) > 0:
                                test_used.append(w_re.lower())

                    # get words that appear in sets BUT NOT lexicon
                    # lexicon = open("lexicon.manual.all.CSV").read().splitlines()
                    # train_unused = set(train_used) - set(lexicon)
                    # test_unused = set(test_used) - set(lexicon)

                    # get OOV
                    train_used_set = set(train_used)
                    oov_words = [x for x in test_used if x not in train_used_set]  # used words that appear in test but not in train
                    oov = (len(oov_words) / len(test_used)) * 100  # percentage of words in `test` that don't appear in train
                    list_of_oovs.append(oov)

                # get average + print
                oov_avg = mean(list_of_oovs)

                if print_avg == 0:
                    str1 = f'Average OOV ({subset}): '
                    print(f"{str1:<23}{oov_avg:>16}")

                if subset == "all":
                    avg_all.append(oov_avg)
                elif subset == "non":
                    avg_non.append(oov_avg)
                elif subset == "abusive":
                    avg_abusive.append(oov_avg)

            print("") if print_avg == 0 else None  # delineate set results

        # average results
        if print_avg == 1:
            print(f"Average OOV (all):     {mean(avg_all)}")
            print(f"Average OOV (non):     {mean(avg_non)}")
            print(f"Average OOV (abusive): {mean(avg_abusive)}\n")

    print(":)")  # celebrate


if __name__ == '__main__':
    get_oov()
