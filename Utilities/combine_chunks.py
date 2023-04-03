# reassemble datasets from chunks
# TODO: investigate reading from folder instead of reassembling
import pandas as pd
import glob
import os

# user-defined variables
dataset_filename = "train.target+comments.tsv"
header = None   # None if no header, "0" if first line is header

# read chunks into memory
chunks_path = f"../KaggleSVM/data/src/{dataset_filename}/"  # evaluate path to chunked dataset
export_path = f"../KaggleSVM/data/src/{dataset_filename}"  # evaluate path to write reassembled dataset to
dataset_name, dataset_ext = dataset_filename.rsplit(".", 1)
delimiter = "\\\\t" if dataset_ext == "tsv" else ","

list_of_dfs = []
list_of_file_paths = glob.glob(os.path.join(chunks_path, dataset_name))
for f in list_of_file_paths:
    df = pd.read_csv(f, index_col=None, header=header, sep=delimiter)
    list_of_dfs.append(df)

# assemble chunks into whole
assembled_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)

# verification
source_df = pd.read_csv('../KaggleSVM/data/src/large_data/train.csv', header=header)
count_reassembled_df = len(assembled_df.index)
count_source_df = len(source_df.index)

print(f"counts equal?: {count_reassembled_df == count_source_df} "
      f"(reassembled count: {count_reassembled_df}, "
      f"source count: {count_source_df})")
print(f"df.equal?: {assembled_df.equals(source_df)}")

# export reassembled dataframe to file
# assembled_df.to_csv(export_path, index=False)
