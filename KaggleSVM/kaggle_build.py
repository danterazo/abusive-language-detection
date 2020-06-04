# LING-X 490
# This file builds, imports, and exports data
# Dante Razo, drazo
from kaggle_preprocessing import read_data, boost_data, sample_data
import pandas as pd
import numpy as np
import os
import glob


# only import once
def get_train():
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    return read_data(dataset)


# Gets 'n' posts, randomly selected, from the dataset. Then save to `.csv`
def build_random(data, sample_size, repeats=3):
    to_export = []
    # sample + export
    for i in range(0, repeats):
        to_export.append(sample_data(data, sample_size))

    export_data("random", to_export)


def build_boosted(data, manual_boost, sample_size, repeats=3):
    data_file = "train.target+comments.tsv"  # name for verbose prints
    to_export = []

    # sample + export, topic
    boosted_topic_data = boost_data(data, data_file, manual_boost)
    for i in range(0, repeats):
        to_export.append(sample_data(boosted_topic_data, sample_size))

    export_data("topic", to_export)

    # boost + sample + export, wordbank
    boosted_wordbank_data = boost_data(data, data_file)

    for i in range(0, repeats):
        to_export.append(sample_data(boosted_wordbank_data, sample_size))

    export_data("wordbank", to_export)


# save data to `.tsv`, `.csv`, etc.
def export_data(source, data, extension=".csv"):
    i = 1

    for d in data:
        filepath = os.path.join("../data/kaggle_data", f"train.{source}{i}{extension}")
        d.to_csv(filepath, index=False, header=False)
        i += 1


# generalized version of the above. `.csv`
def export_df(data, sample="no_sample", i="", path="", prefix="", index=True):
    filepath = os.path.join(path, f"{prefix}.{sample}{i}.csv")
    data.to_csv(filepath, index=index, header=True)


# builds one or both
def build_train(choice, topic, repeats, sample_size, verbose):
    """
    choice: choose which sample types to build. "random", "boosted", or "all"
    topic: topic for manual boosting
    """
    train = get_train()

    build_random(train, sample_size, repeats) if choice is "random" or "all" else None
    build_boosted(train, topic, sample_size, repeats) if choice is "boosted" or "all" else None
    print(f"Datasets built.") if verbose else None


def build_manual_lexicon():
    cwd = os.getcwd()

    os.chdir("lexicon_manual")
    files = glob.glob('*.{}'.format("csv")) + glob.glob('*.{}'.format("tsv"))
    dfs = []

    # assumes they're all the same length (551, as was the provided lexicon)
    for filename in files:
        author = filename.split(".")[-2].strip()

        if author == "dante":
            dfs.append(lexicon_dante(filename))
        elif author == "dd":
            dfs.append(lexicon_dd(filename))
        elif author == "schaede":  # schaede
            dfs.append(lexicon_schaede(filename))

    df = pd.concat(dfs, axis=1)  # one big dataframe
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate "word" columns
    df["avg"] = df[df.columns[1:]].mean(axis=1)  # average class columns
    df["var"] = df[df.columns[1:]].var(axis=1)  # variance of class columns

    df["class"] = False
    df.loc[df.avg > 0.6, "class"] = True  # i.e. at least 2 people say its mildly
    export_df(df, sample="all", prefix="lexicon.manual")  # export it too

    os.chdir(cwd)  # return to previous cwd
    return df  # just in case


# manual lexicon helper f()
def lexicon_dante(filename):
    df = pd.read_csv(filename)[["word", "pass2"]]
    df.columns = ["word", "dante"]
    return df


# manual lexicon helper f()
def lexicon_dd(filename):
    df = pd.read_csv(filename, sep='\t', header=0)[["word", "opinion"]]

    class_vec = []
    for x in df["opinion"]:
        manual_class = str(x).lower()

        if manual_class == "very abusive":
            class_vec.append(2)
        elif manual_class == "mildly abusive":
            class_vec.append(1)
        elif manual_class == "not abusive":
            class_vec.append(0)
        else:
            class_vec.append(np.NaN)

    df["dd"] = class_vec
    df = df[["word", "dd"]]
    return df


# manual lexicon helper f()
def lexicon_schaede(filename):
    df = pd.read_csv(filename, header=0).iloc[:, 0:2]
    df.columns = ["word", "opinion"]

    class_vec = []
    for x in df["opinion"]:
        manual_class = str(x).lower()

        if manual_class == "very abusive":
            class_vec.append(2)
        elif manual_class == "mildly abusive":
            class_vec.append(1)
        elif manual_class == "not abusive":
            class_vec.append(0)
        else:
            class_vec.append(np.NaN)

    df["schaede"] = class_vec
    df = df[["word", "schaede"]]
    return df


# import lexicons, format them, and export them
# in `kaggle_build.py` because it isn't dynamic, i.e. the output is the same after every run
def export_lexicons():
    cwd = os.getcwd()
    os.chdir("../data/kaggle_data")

    # read lexicons; Wiegand lexicons are provided and don't need to be built
    base = pd.read_csv(f"lexicon_wiegand/baseLexicon.txt", sep='\t', header=None, names=["word", "class"])
    exp = pd.read_csv(f"lexicon_wiegand/expandedLexicon.txt", sep='\t', header=None, names=["word", "score"])
    rds = build_manual_lexicon()  # our manually-tagged lexicon; based off Wiegand's base lexicon

    # split word and part-of-speech (Wiegand only)
    base_split = [w.split("_") for w in base["word"]]
    exp_split = [w.split("_") for w in exp["word"]]

    # new dfs
    base["part"] = [s[1] for s in base_split]
    base["word"] = [s[0] for s in base_split]
    exp["part"] = [s[1] for s in exp_split]
    exp["word"] = [s[0] for s in exp_split]

    # filter out non-abusive words
    base_abusive = base[base["class"]]["word"]
    exp_abusive = exp[exp["score"] > 0]["word"]  # according to "Introducing a Lexicon of Abusive Words"
    rds_abusive = rds[rds["class"]]["word"]

    # export
    os.chdir("lexicon_wiegand")
    base.to_csv("lexicon.wiegand.base.csv", header=0, index=False)
    exp.to_csv("lexicon.wiegand.expanded.csv", header=0, index=False)
    base_abusive.to_csv("lexicon.wiegand.base.abusive.csv", header=0, index=False)
    exp_abusive.to_csv("lexicon.wiegand.expanded.abusive.csv", header=0, index=False)

    os.chdir("../lexicon_manual")
    rds_abusive.to_csv("lexicon.manual.all.abusive.csv", header=0, index=False)

    os.chdir(cwd)  # go back to previous cwd


""" MAIN """
topic = ["trump"]  # [str]
to_build = "all"  # "all", "random", or "boosted"
# build_train(to_build, topic)
# export_lexicons()
