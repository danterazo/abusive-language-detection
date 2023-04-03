# split large data into smaller chunks
import pandas as pd
import os

# user-defined variables
chunk_size = 100000  # define chunk size. Individual file size must be <50MB to avoid Git LFS. Recommended: 100000
header = None  # None if no header, "0" if first line is header
dataset_filename = "train.trump.tsv"

# evaluate variables
source_path = f"../KaggleSVM/data/src/large_data/{dataset_filename}"  # evaluate path to large dataset
export_path = f"../KaggleSVM/data/src/{dataset_filename}/"  # evaluate path for saved chunks
dataset_name, dataset_ext = dataset_filename.rsplit(".", 1)
delimiter = "\\\\t" if dataset_ext == "tsv" else ","

# create and write chunks
os.mkdir(export_path) if not os.path.exists(export_path) else None

source_df = pd.read_csv(source_path, header=header, engine="python", sep=delimiter)
# source_df = pd.read_table(source_path, header=header, engine="python", sep=delimiter)
for i, c in source_df.groupby(source_df.index // chunk_size):
    c.to_csv(f"{export_path}{dataset_name}_chunk{i}.{dataset_ext}", index=False, engine="python", sep=delimiter)
