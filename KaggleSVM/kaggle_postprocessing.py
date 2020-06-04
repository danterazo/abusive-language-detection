# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_preprocessing import boost_data
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
