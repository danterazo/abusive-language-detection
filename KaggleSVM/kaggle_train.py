# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from kaggle_preprocessing import read_data
from kaggle_postprocessing import percent_abusive
from kaggle_build import build_train as build_datasets
from kaggle_build import export_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from os import path
import os.path

run = 1  # convenient flag at top of file


# TODO: multithreading? each sample -> thread (i.e. 9 running concurrently, 18 for filtering)
# for each fold of each dataset of each sample type, train an SVM
def fit_data(rebuild, samples, analyzer, ngram_range, manual_boost, repeats, verbose, sample_size, calc_pct):
    """
    rebuild (bool):     if TRUE, rebuild + rewrite the following datasets:
    samples ([str]):    three modes: "random", "boosted", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range (()):   tuple containing lower and upper ngram bounds for CountVectorizer
    manual_boost ([str]):   use given array of strings for filtering instead of built-in wordbanks. Or pass `None`
    repeats (int):      controls the number of datasets built per sample type (if `rebuild` is TRUE)
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    calc_pct (bool):    if TRUE, calculate percentage of abusive words in each sample
    """

    build_datasets(samples, manual_boost, repeats, sample_size, verbose) if rebuild else None  # rebuild datasets

    # struct example: [([random1, random2, ..., random_n], "random"), ...]
    all_data = []
    for x in ["random", "topic", "wordbank"]:
        all_data.append((import_data(x), x))

    # choose one or the other sample type if desired
    if samples is "random":
        all_data = all_data[0]
    elif samples is "boosted":
        all_data = all_data[1:2]

    for sample in all_data:  # for each sample type...
        i = 1

        for set in sample[0]:  # for each set...
            data = pd.DataFrame(set)  # first member of tuple is the dataframe
            sample_type = sample[1].capitalize()  # second member of tuple is a string
            print(f"===== {sample_type}-sample: pass {i} =====") if verbose else None

            # Store data as vectors
            X = data["comment_text"]  # initially reversed because it was easier to separate that way
            y = data["class"]

            # Model pipeline
            print("Instantiating model pipeline (CV & SVM)...") if verbose else None
            vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
            svc = SVC(C=1000, kernel="rbf", gamma=0.001)  # GridSearch best params
            clf = Pipeline([('vect', vec), ('svm', svc)])

            # Testing + results
            k = 5  # number of folds

            filepath = os.path.join("output/pred/", f"pred.{sample_type.lower()}{i}.csv")

            if path.exists(filepath):
                print(f"Importing {sample_type}-sample SVM predictions...") if verbose else None
                y_pred = pd.read_csv(filepath)  # import if `y_pred` has already been computed
                print(f"Data imported!") if verbose else None
            else:
                print(f"Fitting CountVectorizer & training {sample_type}-sample SVM...") if verbose else None
                y_pred = cross_val_predict(clf, X, y, cv=k, n_jobs=14)  # else, compute
                pd.DataFrame(y_pred).to_csv(filepath, index=False)  # save preds
                print(f"SVM trained!") if verbose else None

            # calculate % abusive (multithreaded)
            if calc_pct:
                print(f"\nCalculating abusive content percentage(s)...\n")
                pct = percent_abusive(data)
                print(pct)
                # print(f"{pct[1]}% abusive (manual lexicon)") # debugging
                export_df(pct, sample_type, i, path="output/stats/", prefix="percent")

            # report results + export
            report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
            print(f"\nClassification Report[{sample_type.lower()}, {analyzer}, ngram_range{ngram_range}]:\n{report}\n")
            export_df(report, sample_type, i, path="output/report", prefix="report")
            i += 1


def import_data(sample_type):
    to_return = []

    for i in range(1, 4):
        to_return.append(read_data(f"train.{sample_type}{i}.csv", verbose=False))

    return to_return


""" SCRIPT CONFIG """
samples = "all"  # "random", "boosted_topic", "boosted_wordbank", or "all"
analyzer = "word"  # "char" or "word"
ngram_range = (1, 3)  # int 2-tuple / couple
manual_boost = ["trump"]  # ["trump"]  # None, or an array of strings
rebuild = False  # rebuild datasets + export
repeats = 3  # number of datasets per sample type
verbose = True  # suppresses prints if FALSE
calc_pct = True  # calculate abusive example percentage per sample
sample_size = 20000

""" MAIN """
# need main for multithreaded boosting
if __name__ == '__main__':
    if run is 1:
        fit_data(rebuild, samples, analyzer, ngram_range, manual_boost, repeats, verbose, sample_size, calc_pct)
