# LING-X 490: Abusive Language Detection
# This file trains SVMs on uniquely-sampled datasets
# Dante Razo, drazo, 2020-05-15
from os import path

from kaggle_postprocessing import calc_pct_abusive
from kaggle_preprocessing import read_data
from kaggle_build import build_train as build_datasets
from kaggle_build import export_lexicons as build_lexicons
from kaggle_build import export_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd

run = 1  # convenient flag at top of file


# for each fold of each dataset of each sample type, train an SVM
def fit_data(rebuild, samples, analyzer, ngram_range, manual_boost, repeats, verbose, sample_size, calc_pct):
    """
    rebuild (bool):     if TRUE, rebuild + rewrite the following datasets:
    samples ([str]):    three modes: "random", "boosted", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range ((int,int)):   tuple containing lower and upper ngram bounds for CountVectorizer
    manual_boost ([str]):   use given list of strings for filtering instead of built-in wordbanks. Or pass `None`
    repeats (int):      controls the number of datasets built per sample type (if `rebuild` is TRUE)
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    calc_pct (bool):    if TRUE, calculate percentage of abusive words in each sample
    """

    # rebuild datasets
    if rebuild:
        build_datasets(samples, manual_boost, repeats, sample_size, verbose)
        build_lexicons()

    # struct example: [([random1, random2, ..., random_n], "random"), ...]
    all_data = []
    for x in ["random", "topic", "wordbank"]:
        all_data.append((import_data(x, repeats), x))

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

            # calculate + export predictions
            y_pred = pred_helper(X, y, clf, k, sample_type, i)

            # calculate % abusive
            pct_helper(data, sample_type, i) if calc_pct else None

            # report results + export
            report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
            print(f"\nClassification Report[{sample_type.lower()}, {analyzer}, ngram_range{ngram_range}]:\n{report}\n")
            export_df(report, sample_type, i, path="output/report", prefix="report")
            i += 1


def import_data(sample_type, n):
    to_return = []

    for i in range(1, n + 1):
        to_return.append(read_data(f"train.{sample_type}{i}.csv", verbose=False))

    return to_return


def pred_helper(x, y, clf, k, sample_type, i):
    X = x  # as it should be

    pred_path = path.join("output/pred/", f"pred.{sample_type.lower()}{i}.csv")
    if path.exists(pred_path):
        print(f"Importing {sample_type}-sample SVM predictions...") if verbose else None
        y_pred = pd.read_csv(pred_path)  # import if `y_pred` has already been computed
        print(f"Data imported!") if verbose else None
    else:
        print(f"Fitting CountVectorizer & training {sample_type}-sample SVM...") if verbose else None
        y_pred = cross_val_predict(clf, X, y, cv=k, n_jobs=14)  # else, compute
        pd.DataFrame(y_pred).to_csv(pred_path, index=False)  # save preds
        print(f"SVM trained!") if verbose else None

    return y_pred


def pct_helper(data, sample_type, i):
    pct_path = path.join("output/stats/percent_abusive", f"percent.{sample_type.lower()}{i}.csv")
    if path.exists(pct_path):
        print(f"\nImporting {sample_type}-sample abusive content percentages...")
        pct = pd.read_csv(pct_path)  # import if already computed
        print(f"Percentages Imported!")
    else:
        print(f"\nCalculating {sample_type}-sample abusive content percentages...")
        pct = calc_pct_abusive(data)  # else, calculate
        print(f"Percentages calculated!")

    print(f"{pct}")
    export_df(pct, sample_type.lower(), i, path="output/stats/percent_abusive", prefix="percent", index=False)


""" MAIN """


# separate main to protect variable names in inner scope
def main():
    if run is 1:
        samples = "all"  # "random", "boosted_topic", "boosted_wordbank", or "all"
        analyzer = "word"  # "char" or "word"
        ngram_range = (1, 3,)  # int 2-tuple (couple)
        manual_boost = ["trump"]  # ["trump"]  # None, or a list of strings
        rebuild = False  # rebuild datasets + export
        repeats = 3  # number of datasets per sample type
        verbose = True  # suppresses prints if FALSE
        calc_pct = True  # calculate abusive example percentage per sample
        sample_size = 20000

        fit_data(rebuild, samples, analyzer, ngram_range, manual_boost, repeats, verbose, sample_size, calc_pct)


# need main for future multithreaded function calls
if __name__ == '__main__':
    main()
