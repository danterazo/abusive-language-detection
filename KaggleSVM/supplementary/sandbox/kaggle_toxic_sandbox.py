import pandas as pd

toxic = pd.read_csv("../../data/src_new/kaggle-toxic_train-V2.CSV")  # read kaggle_toxic dataset

""" Build toxic V2
toxic = pd.read_csv("data/src_new/kaggle-toxic_train-clean.CSV")  # read kaggle_toxic dataset
toxic["class"] = toxic[["toxic", "severe_toxic"]].max(axis=1)  # if it's toxic or severely toxic, class = 1

# (number of abusive examples, i.e. class=1) / (number of examples total)
percent_abusive = np.round(len(toxic[toxic["class"] == 1]) / len(toxic) * 100, 2)

# export dataset
toxic = toxic.drop(["toxic","severe_toxic","obscene","threat","insult","identity_hate"], axis=1)
toxic.to_csv("data/src_new/kaggle-toxic_train-V2.CSV", index=False)

print(f"percent abusive: {percent_abusive}")
"""

"""
# count # of trump comments in original
# toxic_trump = boost_data(toxic, "toxic", manual_boost=["trump"], verbose=True)

# Count most common words in kaggle_toxic abusive posts
stop = stopwords.words("english")
toxic["nostop"] = toxic["comment_text"].apply(lambda x: " ".join([word.replace("\"", "") for word in x.split() if word.lower() not in stop]))

abusive_only = toxic[toxic["class"] == 1]
toxic_count = Counter(" ".join(abusive_only["nostop"]).split()).most_common(200)
print(toxic_count)
"""

# count number of abusive entries per lexicon
base = pd.read_csv("../../data/lexicon_wiegand/baseLexicon.txt", delimiter="\t", names=["word", "class"])
expanded = pd.read_csv("../../data/lexicon_wiegand/expandedLexicon.txt", delimiter="\t", names=["word", "class"])
manual = pd.read_csv("../../data/lexicon_manual/lexicon.manual.all.csv", header=0)

base_abusive = base[base['class']]
exp_abusive = expanded[expanded['class'] > 0]
manual_abusive = manual[manual['class']]

print(f"manual: {len(manual_abusive)}")
print(f"base: {len(base_abusive)}")
print(f"expanded: {len(exp_abusive)}")

non_abusive = data[data["class"] == 0]
abusive_only = data[data["class"] == 1]
