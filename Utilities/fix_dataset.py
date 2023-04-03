import pandas as pd

# TODO: Fix this. It doesn't work with `train.target+comments.tsv` or `train.trump.tsv`. Formatting issues.

# user-defined variables
dataset_filename = "train.target+comments.tsv"
header = None  # None if no header, "0" if first line is header

# evaluate variables
dataset_name, dataset_ext = dataset_filename.rsplit(".", 1)
import_path = f"../KaggleSVM/data/src/large_data/{dataset_filename}"  # evaluate path to chunked dataset
export_path = f"../KaggleSVM/data/src/{dataset_name}.csv"  # evaluate path to write reassembled dataset to

# import data and clean
data_list = []  # temporary; used for constructing dataframe
with open(import_path, "r", encoding="utf-8") as d:
    lines = d.readlines()

    for x in lines:
        if dataset_ext == ".TSV":
            split_line = x.split(maxsplit=1)
        else:  # default: CSV
            split_line = x.split(",", maxsplit=1)

        if len(split_line) == 2:  # if len != 2, there's no score, so throw the example out
            # data_list.append([float(split_line[0].strip()), str(split_line[1])])  # original example
            data_list.append([str(split_line[0].strip()), str(split_line[1])])

df = pd.DataFrame(data_list, columns=["score", "comment_text"])

# write data
df.to_csv(export_path, index=False)
df.to_csv(f"../KaggleSVM/data/src/{dataset_name}.tsv", index=False, sep="\t")
