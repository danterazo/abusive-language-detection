import pandas as pd

# Sandra 2020-10-02: extract first 1000 abusive posts from the large set
# data = pd.read_csv("data/src/train.target+comments.TSV", sep="\t")
data = pd.DataFrame(columns=["comment", "class"])
with open("data/src/train.target+comments.TSV", encoding="utf-8") as f:
    threshold = 0.5

    for x in f.readlines():
        split = x.split("\t", maxsplit=1)

        if len(split) == 2:
            score, comment = spli
            data = data.append([comment, (float(score) >= threshold)])
        else:
            print(comment)

print(data.head())
data.to_csv("data/src/train.class+comments.CSV")
