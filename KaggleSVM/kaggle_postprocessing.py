# LING-X 490
# This file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_preprocessing import boost_data
from multiprocessing import Process, Queue
import pandas as pd


# calculate % examples in given data that contains abusive words. returns df
def percent_abusive(data):
    """
    data (df): dataframe to filter
    lex (str): lexicon to filter with. Either "we" (wiegand extended) or "rds" (our manually tagged dataset)
    """
    jobs = []
    to_return = []  # [[source (str), percent (float)], ...]
    q = Queue()

    filenames = ["data/lexicon_manual/lexicon.manual.all.abusive.csv",
                 "data/lexicon_wiegand/lexicon.wiegand.base.abusive.csv",
                 "data/lexicon_wiegand/lexicon.wiegand.expanded.abusive.csv"]

    for f in filenames:
        f_split = f.split(".", 3)
        source = f_split[1] + "." + f_split[2]

        boost_list = open(f).read().splitlines()
        p = Process(target=boost_multithreaded, args=(data, source, boost_list, q,))
        jobs.append(p)
        p.start()

    # multithreaded boosting; waits for all jobs to finish
    for process in jobs:
        process.join()

    # get output
    for x in jobs:
        curr = q.get()
        boosted_df, name = curr
        pct = round(len(boosted_df) / len(data) * 100, 2)

        to_return.append([pct, name])

    # return dataframe
    to_return = pd.DataFrame(to_return, columns=["percent_abusive", "source"])
    return to_return


# multiprocess `boost_data()`
def boost_multithreaded(data, source, manual_boost, queue):
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

    queue.put(boosted_data)
    return boosted_data, source
