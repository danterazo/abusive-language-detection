# LING-X 490
# This standalone file takes built data and reformats / averages / analyzes it
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

    filename = "../data/kaggle_data/lexicon_manual/lexicon.manual.all.abusive.csv"
    lexicon_rds = pd.read_csv(filename, sep=",", header=0)
    boost_list = list(lexicon_rds)
    p1 = Process(target=boost_multithreaded, args=(data, "manual", boost_list, q,))
    jobs.append(p1)
    p1.start()

    filename = "../Data/kaggle_data/lexicon_wiegand/lexicon.wiegand.base.abusive.csv"
    lexicon_wiegand_base = pd.read_csv(filename, sep=",", header=0)
    boost_list = list(lexicon_wiegand_base)
    p2 = Process(target=boost_multithreaded, args=(data, "Wiegand (Base)", boost_list, q,))
    jobs.append(p2)
    p2.start()

    filename = "../Data/kaggle_data/lexicon_wiegand/lexicon.wiegand.expanded.abusive.csv"
    lexicon_wiegand_exp = pd.read_csv(filename, sep=",", header=0)
    boost_list = list(lexicon_wiegand_exp)
    p3 = Process(target=boost_multithreaded, args=(data, "Wiegand (Expanded)", boost_list, q,))
    jobs.append(p3)
    p3.start()

    # multithreaded boosting; waits for all jobs to finish
    for process in jobs:
        process.join()

    # get output
    for x in jobs:
        curr = q.get()
        boosted_df, name = curr
        pct = round(len(boosted_df) / len(data) * 100, 2)

        to_return.append([pct, name])

    to_return = pd.DataFrame(to_return, columns=["percent_abusive", "source"])
    print(to_return)  # DEBUG
    return to_return


# multiprocess `boost_data()`
def boost_multithreaded(data, source, manual_boost, queue):
    boosted_data = boost_data(data, "", False, manual_boost)

    queue.put(boosted_data)
    return boosted_data, source
