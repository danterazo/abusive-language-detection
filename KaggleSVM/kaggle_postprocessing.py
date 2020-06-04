# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_preprocessing import boost_data
from multiprocessing import Process, Queue
import pandas as pd

""" GLOBAL VARIABLES for MULTIPROCESSING"""
boost_queue = Queue()


# calculate % examples in given data that contains abusive words. returns df
def percent_abusive(data):
    results_df = pd.DataFrame(columns=["pct_abusive", "source_lexicon"])

    filenames = ["data/lexicon_manual/lexicon.manual.all.abusive.csv",
                 "data/lexicon_wiegand/lexicon.wiegand.base.abusive.csv",
                 "data/lexicon_wiegand/lexicon.wiegand.expanded.abusive.csv"]

    for f in filenames:
        f_split = f.split(".", 3)
        source = f_split[1] + "." + f_split[2]

        boost_list = open(f).read().splitlines()
        boosted_df = boost_data(data, "", False, manual_boost=boost_list)
        pct = round(len(boosted_df) / len(data) * 100, 2)

        results_df.loc[len(results_df)] = [pct, source]

    return results_df


# multiprocess `boost_data()`
def boost_multithreaded(data, source, manual_boost, boost_queue):
    """
    - Params
        - data (df): dataframe to boost
        - source (str): lexicon name
        - manual_boost ([str]): wordbank derived from the `source` lexicon
        - queue (multiprocessing.Queue object): used to run all boosting jobs concurrently
    - Return
        - Tuple containing the boosted data and a name: (df, str,)
    """

    boosted_data = boost_data(data, "", False, manual_boost)

    print(f"len boosted: {len(boosted_data)}")
    # boost_queue.put((1, 1))
    boost_queue.put(boosted_data)
    print(f"added to queue: {boost_queue}")  # debugging, to remove
    return boosted_data, source
