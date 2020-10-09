# LING-X 490
# This file imports, formats, and processes data for use in SVM
# Dante Razo, drazo
import re

import pandas as pd


# read + process training data (format: [class, comment])
def read_data(dataset, verbose=True):
    data_dir = "data"  # common directory for all datasets

    print(f"Importing `{dataset}`...") if verbose else None  # progress indicator
    data_list = []  # temporary; used for constructing dataframe
    extension = dataset[-4:]

    # import data
    with open(f"{data_dir}/{dataset}", "r", encoding="utf-8") as d:
        entries = d.readlines()

        for e in entries:
            if extension == ".TSV":
                split_line = e.split("\t", 1)
            else:  # default: CSV
                split_line = e.split(",", 1)

            if len(split_line) is 2:  # else: there's no score, so throw the example out
                data_list.append([float(split_line[0].strip()), str(split_line[1])])

    data = pd.DataFrame(data_list, columns=["score", "comment_text"])
    print(f"Data {data.shape} imported!\n") if verbose else None  # progress indicator

    kaggle_threshold = 0.50  # from Kaggle documentation

    # create class vector
    data["class"] = 0
    data.loc[data.score >= kaggle_threshold, "class"] = 1

    # remove score vector; replaced by class (bool) vector
    data = data.drop(columns="score")

    # swap column/feature order
    data = data[["class", "comment_text"]]
    return data


# given kaggle toxicity data, format it to match the others
def format_kaggle_toxicity():
    kaggle_toxic = pd.read_csv("data/src_new/kaggle-toxic_train.CSV",
                               names=["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

    kaggle_toxic = kaggle_toxic.drop("id", axis=1)
    kaggle_toxic = kaggle_toxic.iloc[1:]  # remove first row (old headers/names)
    kaggle_toxic.to_csv("data/src_new/kaggle-toxic_train-clean.CSV", index=0)


# shuffle data then sample from it
def sample_data(data, size):
    return data.sample(frac=1)[0:size]


# boosts data, i.e. returns data that contains any word in given wordbank
def boost_data(data, data_name, manual_boost=[], verbose=True):
    """
    data (df):          dataset to filter
    topics ([str]]):    word(s) to filter with. this wordbank bypasses the banks below
    """
    manual_boost_len = len(manual_boost)
    if verbose:
        if manual_boost_len > 0:
            print(f"Boosting `{data_name}` on custom wordbank of size {manual_boost_len}...")
        else:
            print(f"Boosting `{data_name}` on built-in wordbank...")

    # source (built upon): https://dictionary.cambridge.org/us/topics/religion/islam/d
    islam_wordbank = ["allah", "caliphate", "fatwa", "hadj", "hajj", "halal", "headscarf", "hegira", "hejira",
                      "hijab", "islam", "islamic", "jihad", "jihadi", "mecca", "minaret", "mohammeden", "mosque",
                      "muhammad", "mujahideen", "muslim", "prayer", "mat", "prophet", "purdah", "ramadan", "salaam",
                      "sehri", "sharia", "shia", "sunni", "shiism", "sufic", "sufism", "suhoor", "sunna", "koran",
                      "qur'an", "yashmak", "ISIS", "ISIL", "al-Qaeda", "Taliban"]

    # source: https://www.usatoday.com/story/news/2017/03/16/feminism-glossary-lexicon-language/99120600/
    metoo_wordbank = ["metoo", "feminism", "victim", "consent", "patriarchy", "sexism", "misogyny", "misandry",
                      "misogynoir", "cisgender", "transgender", "transphobia", "transmisogyny", "terf", "swef",
                      "non-binary", "woc", "victim-blaming", "trigger", "privilege", "mansplain", "mansplaining",
                      "manspread", "manspreading", "woke", "feminazi"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#Politics_and_economics
    politics_wordbank = ["republican", "GOP", "democrats", "liberal", "liberals", "abortion", "brexit",
                         "anti-semitism", "atheism", "conservatives", "capitalism", "communism", "Cuba", "fascism",
                         "Fox News", "immigration", "kashmir", "harambe", "israel", "hitler", "mexico",
                         "neoconservatism", "neoliberalism", "palestine", "9/11", "socialism", "Clinton", "Trump",
                         "Sanders", "Guantanamo", "torture", "Flight 77", "Marijuana", "sandinistas"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#History
    history_wordbank = ["Apartheid", "Nazi", "Black Panthers", "Rwandan Genocide", "Jim Crow", "Ku Klux Klan"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#Religion
    religion_wordbank = ["jew", "judaism", "christian", "christianity", "Jesus Christ", "Baptist", "WASP",
                         "Protestant", "Westboro Baptist Church"]

    # source: Sandra's suggestions, email from 2020-03-16
    sandra_wordbank = ["trump", "obama", "trudeau", "clinton", "hillary", "donald", "tax", "taxpayer", "vote",
                       "voting", "election", "party", "president", "politician", "women", "woman", "fact",
                       "military", "citizen", "nation", "church", "christian", "muslim", "liberal", "democrat",
                       "republican", "religion", "religious", "administration", "immigrant", "gun", "science",
                       "freedom", "solution", "corporate"]

    # words with special capitalization rules; except from capwords() function call below
    special_caps = ["al-Qaeda", "CNN", "KKK", "LGBT", "LGBTQ", "LGBTQIA"]

    # manually observed abusive words in explicit examples
    explicitly_abusive = ["sh*tty"]

    if manual_boost_len > 0:
        # use the given topics (arg)
        combined_topics = set(manual_boost)
    else:
        # else, combine the built-in wordbanks
        combined_topics = set(islam_wordbank + metoo_wordbank + politics_wordbank + history_wordbank + religion_wordbank + \
                              sandra_wordbank + special_caps + explicitly_abusive)

    wordbank = combined_topics  # easy toggle if you want to focus on a specific topic instead

    # wordbank = wordbank + ["#" + word for word in topic]  # ...then add hashtags for all words
    wordbank_regex = re.compile("|".join(wordbank), re.IGNORECASE)  # compile regex. case insensitive

    # idea: .find() for count. useful for threshold
    filtered_data = data[data["comment_text"].str.contains(wordbank_regex)]
    print(f"Data boosted to size {filtered_data.shape[0]}.\n") if verbose else None
    return filtered_data


if __name__ == '__main__':
    # format_kaggle_toxicity()
    pass
