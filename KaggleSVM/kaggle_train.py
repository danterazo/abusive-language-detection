# LING-X 490: Abusive Language Detection Research
# This file trains SVMs on uniquely-sampled datasets
# Dante Razo, drazo, 2020-05-15
from os import path

from kaggle_postprocessing import calc_pct_abusive
from kaggle_preprocessing import read_data, boost_data
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
def fit_data(rebuild, samples, analyzer, ngram_range, manual_boost, per_sample, verbose, sample_size, calc_pct, decimals):
    """
    rebuild (bool):     if TRUE, rebuild + rewrite the following datasets:
    samples ([str]):    three modes: "random", "boosted", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range ((int,int)):    tuple containing lower and upper ngram bounds for CountVectorizer
    manual_boost ([str]):       use given list of strings for filtering instead of built-in wordbanks. Or pass `None`
    per_sample (int):           controls the number of datasets built per sample type (if `rebuild` is TRUE)
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    calc_pct (bool):    if TRUE, calculate percentage of abusive words in each sample
    decimals (int):     number of decimals to round percentages to
    """

    # rebuild datasets
    if rebuild:
        build_datasets(samples, manual_boost, per_sample, sample_size, verbose)
        build_lexicons()

    # struct example: [([random1, random2, ..., random_n], "random"), ...]
    all_data = []
    for x in ["random", "topic", "wordbank"]:
        all_data.append((import_data(x, per_sample), x))

    # choose one or the other sample type if desired
    if samples is "random":
        all_data = all_data[0]
    elif samples is "boosted":
        all_data = all_data[1:2]

    for sample in all_data:  # for each sample type...
        i = 1
        reports_to_avg = []  # list of reports to soon be averaged
        sample_type = ""  # sample type name (e.g. "random", "boosted", etc.) in outer scope for preservation

        for set in sample[0]:  # for each set...
            data = pd.DataFrame(set)  # first member of tuple is the dataframe
            sample_type = sample[1].lower()  # second member of tuple is a string
            print(f"===== {sample_type.capitalize()}-sample: pass {i} =====") if verbose else None

            # store data as vectors
            X = data["comment_text"]
            y = data["class"]

            # model pipeline
            print("Instantiating model pipeline (CV & SVM)...") if verbose else None
            vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
            svc = SVC(C=1000, kernel="rbf", gamma=0.001)  # GridSearch best params
            clf = Pipeline([('vect', vec), ('svm', svc)])

            # cross-validation
            k = 5

            # calculate + export predictions
            y_pred = pred_helper(X, y, clf, k, sample_type, i, verbose)

            # calculate % abusive
            pct_helper(data, sample_type, i, decimals, verbose) if calc_pct else None

            # report results + export
            report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
            # report = round_report_df(report_to_percentage(report), decimals)  # convert precision + recall columns to percentages + round

            print(f"\nClassification Report[{sample_type}, {analyzer}, ngram_range{ngram_range}]:\n{report}\n")
            export_df(report, sample_type, i, folder="output/report", prefix="report")
            reports_to_avg.append(report)

            bin_data(data, sample_type, i, analyzer, ngram_range)

            i += 1

        # average all reports of the same sample type (e.g. random1, random2, random3)
        print(f"===== {sample_type}-sample: Average of {len(reports_to_avg)} =====") if verbose else None
        averaged = pd.concat(reports_to_avg).groupby(level=0).mean()  # given a list of dataframes, average their values
        # averaged = round_report_df(averaged, decimals)  # convert precision + recall columns to percentages + round
        export_df(averaged, sample_type, i=".avg", folder="output/report", prefix="report")  # export the averaged report
        print(f"\nClassification Report[{sample_type}, {analyzer}, ngram_range{ngram_range}]:\n{averaged}\n")


# given data, sort it into explicitly abusive and implicitly abusive datasets
def bin_data_helper(df):
    data_abusive = df[df["class"] == 1]  # filter data to only abusive examples, i.e. discard non-abusive ones
    explicit_list = open("data/lexicon_manual/lexicon.manual.all.explicit.CSV").read().splitlines()  # list of explicitly abusive words

    data_explicit = boost_data(data_abusive, "", False, manual_boost=explicit_list)
    data_implicit = pd.concat([data_abusive, data_explicit]).drop_duplicates(keep=False)

    return data_explicit, data_implicit


# binning experiment
def bin_data(data_with_preds, sample_type, i, analyzer, ngram_range):
    # split data into explictly abusive and implictly abusive
    data_with_preds["pred"] = pd.read_csv(path.join("output/pred", f"pred.{sample_type.lower()}{i}.CSV"))  # add preds as new column

    explicit_data, implicit_data = bin_data_helper(data_with_preds)

    # store data as vectors
    y_explicit = explicit_data["class"]
    y_pred_explicit = explicit_data["pred"]
    y_implicit = implicit_data["class"]
    y_pred_implicit = implicit_data["pred"]

    # reports
    report_explicit = pd.DataFrame(classification_report(y_explicit, y_pred_explicit, output_dict=True, zero_division=0)).transpose()
    report_implicit = pd.DataFrame(classification_report(y_implicit, y_pred_implicit, output_dict=True, zero_division=0)).transpose()

    # print + export
    print(f"\nClassification Report[{sample_type}.explicit, {analyzer}, ngram_range{ngram_range}]:\n{report_explicit}\n")
    export_df(report_explicit, sample_type, f"{i}.explicit", folder="output/report/binning", prefix="report")

    print(f"\nClassification Report[{sample_type}.implicit, {analyzer}, ngram_range{ngram_range}]:\n{report_implicit}\n")
    export_df(report_implicit, sample_type, f"{i}.implicit", folder="output/report/binning", prefix="report")


# given a report dataframe, convert floats to percentages
def report_to_percentage(report):
    report.iloc[:, [0, 1]] = report.iloc[:, [0, 1]] * 100  # float -> percentage on select columns (precision, recall)
    return report


# given a report dataframe, round it
def round_report_df(report, decimals):
    report.iloc[:, [0, 1]] = report.iloc[:, [0, 1]].round(decimals)  # round select columns (precision, recall)

    return report


# read data into dataframes, then return a list of all those dataframes
def import_data(sample_type, n):
    to_return = []

    for i in range(1, n + 1):
        to_return.append(read_data(f"train.{sample_type}{i}.CSV", verbose=False))

    return to_return


# get predictions
def pred_helper(x, y, clf, k, sample_type, i, verbose):
    pred_path = path.join("output/pred/", f"pred.{sample_type.lower()}{i}.CSV")
    if path.exists(pred_path):
        print(f"Importing {sample_type}-sample SVM predictions...") if verbose else None
        y_pred = pd.read_csv(pred_path)  # import if `y_pred` has already been computed
        print(f"Data imported!") if verbose else None
    else:
        print(f"Fitting CountVectorizer & training {sample_type}-sample SVM...") if verbose else None
        y_pred = cross_val_predict(clf, x, y, cv=k, n_jobs=14)  # ...else, compute predictions
        pd.DataFrame(y_pred).to_csv(pred_path, index=False)  # save preds
        print(f"SVM trained!") if verbose else None

    return y_pred


# get percent abusive
def pct_helper(data, sample_type, i, decimals, verbose):
    pct_path = path.join("output/stats/percent_abusive", f"percent.{sample_type.lower()}{i}.CSV")
    if path.exists(pct_path):
        print(f"\nImporting {sample_type}-sample abusive content percentages...") if verbose else None
        pct = pd.read_csv(pct_path)  # import if already computed
        print(f"Percentages Imported!") if verbose else None
    else:
        print(f"\nCalculating {sample_type}-sample abusive content percentages...") if verbose else None
        pct = calc_pct_abusive(data, decimals, verbose)  # else, calculate
        print(f"Percentages calculated!") if verbose else None

    print(f"{pct}")
    export_df(pct, sample_type.lower(), i, folder="output/stats/percent_abusive", prefix="percent", index=False)


# separate Main to protect variable names in inner scope
def main():
    if run is 1:
        samples = "all"  # "random", "boosted_topic", "boosted_wordbank", or "all"
        analyzer = "word"  # "char" or "word"
        ngram_range = (1, 3,)  # int 2-tuple (couple)
        manual_boost = ["trump"]  # None, or a list of strings
        rebuild = False  # rebuild datasets + export
        repeats = 3  # number of datasets per sample type
        verbose = True  # suppresses prints if FALSE
        calc_pct = True  # calculate abusive percentages
        decimals = 2  # number of decimals to round percentages to (e.g. abusive example percentages, in classification reports, etc.)
        sample_size = 20000

        fit_data(rebuild, samples, analyzer, ngram_range, manual_boost, repeats, verbose, sample_size, calc_pct, decimals)


# need main for future multithreaded function calls
if __name__ == '__main__':
    main()
