# WIP
This code is a mess but it won't be for long. Check back in September!

TODO: git LFS pull instructions / setup

# Introduction & Research Goals 
The goal of this project is to improve abusive language detection with a focus on implicit abuse. **Python** was used for data preprocessing, dataset builds, and SVM training. **R** was used to verify dataset properties (e.g. length, headers, etc.). The paper was written in and compiled from **LaTeX**.

## Paper
The paper is a WIP but will include detailed research methods, results, citations, and more.

Once the paper is accepted, see `Paper/` for a compiled PDF, uncompiled LaTeX, and assets.

The plan is to submit this paper to the [_Fourth Workshop on Online Abuse and Harms_](https://www.aclweb.org/portal/content/fourth-workshop-online-abuse-and-harms) (WOAH), co-located with the 2020 conference on _Empirical Methods in Natural Language Processing_ (EMNLP 2020).

## Results
Work in progress. Stay tuned!

# Code
This repository contains all of the resources you will need to replicate results. Boosting data takes an absurd amount of time depending on the lexicon used; I recommend using a computer with at least **4c/4t** and **16GB** of memory. This is merely a recommendation and not a requirement.

## Running the script
1. Install dependencies
    - **Vanilla Python**: `pip3 install -r requirements.txt`
    - **Conda**:
      1. Select the desired conda `env` before installing (see `nlpGPU_env.yml` for my NLP-focused conda env)
      2. `conda install --file requirements.txt`

2. Clone
- `git clone https://github.com/danterazo/abusive-language-detection.git`
  
3. Configure
    - See the `main()` in `kaggle_train.py`
        | Variable     | Data Type    | Default Value | Possible Values                                      | Purpose                                                      |
        | ------------ | ------------ | ------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
        | `samples`      | *str*        | **"all"**     | "random", "boosted_topic", "boosted_wordbank", "all" | Lets the user choose which sample types to train on.         |
        | `analyzer`     | *str*        | **"word"**    | "char", "word"                                       | Toggle n-gram analyzer.                                |
        | `ngram_range`  | *(int, int)* | **(1,3)**    | {i &#124; i &isin; Z<sup>+</sup>}                     | Couple (2-tuple) of lower and upper n-gram boundaries.           |
        | `manual_boost` | *[str]*      | **["trump"]** | a list of strings OR **None**          | If not **None**, override predefined wordbanks when boosting. |
        | `rebuild`      | *bool*       | **False**     | True, False                                          | If **True**, resample + rebuild training data and lexicons. The former is computationally expensive. |
        | `repeats`      | *int*        | **3**         | {i &#124; i &isin; Z<sup>+</sup>, i > 0}             | Set the number of each sample type to build and train. Ignored if `rebuild` is **False**. |
        | `sample_size`  | *int*        | **20000**     | {i &#124; i &isin; Z<sup>+</sup>, i > 0}             | Set the size of each dataset when building. If any set has <2000 examples, the others will be trimmed to match it. Ignored if `rebuild` is **False**. |
        | `verbose`      | *bool*       | **True**      | True, False                                          | Controls verbose print statements. Passed to other functions like a react prop. |
        | `calc_pct`     | *bool*       | **True**      | True, False                                          | If **True**, calculate the percentage of abusive words in each sample. Uses *manual*, *Wiegand Base*, and *Wiegand Extended* lexicons. Very computationally expensive. |
    
4. Train
    - Once you've configured the script, simply run `kaggle_train.py`. No user input is required.
    - `python3 kaggle_train.py`

5. Wait patiently for results
    - Percentage calculation (see `calc_pct` above) is time-consuming due to regex compilation and boosting step
        - TODO: run in parallel (multithreaded)
    - **Rebuilt datasets** can be found in `data/`
    - **Class predictions** can be found in `output/pred/`
    - **Classification reports** can be found in `output/report/`

## Nomenclature
Throughout the code I refer to our manually-tagged lexicon, based off Wiegand's base lexicon, as either *manualLexicon*
or *rds*, the latter being the initials of the contributors' last names (Dante Razo, DD, Leah Schaede).


# Files
## kaggle_build.py
This file exports data for later use. Included in the repo is prebuilt data, so its not necessary to run this script.

To run, set the `rebuild` flag in `kaggle_train.py` to **TRUE** then run the latter file.

### `build_train()`
Builds and exports sampled training sets from large `train.CSV` dataset
- Params
    - `choice` (*str*): choose which sample types to build. "random", "boosted", or "all"
    - `topic` (*[str]*): list of strings to boost on
    - `repeats` (*int*): number of datasets to build per sample type
- Return
    - None
- Write
    - None

#### `get_train()`
Quick function to import `train.target+comments.TSV` + call `kaggle_preprocessing.read_data()` to format it.
- Params
    - None
- Return
    - Preprocessed full training dataset: (*df*)
- Write
    - None

#### `build_random()`
Call `kaggle_preprocessing.sample_data()` to shuffle + cut `train.CSV` down to desired sample size.
- Params
    - `data` (*df*): full training data to sample from
    - `sample_size` (*int*): upper bound for cutting result down to size
    - `repeats` (*int*): number of datasets to build per sample type
- Return
    - None
- Write
    - `data/train.random{i}.CSV` for index `i`
    

#### `build_boosted()`
Call `kaggle_preprocessing.boost_data()` to boost `train.target+comments.TSV` on built-in wordbank or user-defined
wordbank (passed as param `manual_boost`).
- Params
    - `data` (*df*): full training data to sample from
    - `manual_boost` (*[str]*): list of strings to boost on
    - `sample_size` (*int*): upper bound for cutting result down to size
    - `repeats` (*int*): number of datasets to build per sample type
- Return
    - None
- Write
    - `data/train.boosted{i}.CSV` for index `i`


### `export_lexicons()`
Wrapper function for importing lexicons. Reformats them accordingly as well; this could be considered processing but I
left it in `kaggle_build.py` because it also exports them.
- Params
    - None
- Return
    - None
- Write
    - `data/lexicon_wiegand/`
        - `lexicon.wiegand.base.CSV`
        - `lexicon.wiegand.expanded.CSV`
        - `lexicon.wiegand.base.explicit.CSV`
        - `lexicon.wiegand.expanded.explicit.CSV`
    - `data/lexicon_manual/`
        - `lexicon.manual.all.explicit.CSV`

#### `build_manual_lexicon()`
Another wrapper function. This calls helper functions to import and process the manually-tagged lexicons. Finally,
it combines them into one DataFrame and exports it.
- Params
    - None
- Return
    - None
- Write
    - `data/manual_lexicon/lexicon.manual.all`

##### `lexicon_dante()`
Strips unnecessary columns from my manually-tagged lexicon.
- Params
    - `filename` (*str*): the name of the csv to be read
- Return
    - Processed lexicon: (*df*)
- Write
    - None

##### `lexicon_dd()`
Strips unnecessary columns from DD's manually-tagged lexicon (`.TSV`), then convert text classes to ints.
- Params
    - `filename` (*str*): the name of the csv to be read
- Return
    - Processed lexicon: (*df*)
- Write
    - None

##### `lexicon_schaede()`
Strips unnecessary columns from Schaede's manually-tagged lexicon (`.CSV`), then convert text classes to ints.
- Params
    - `filename` (*str*): the name of the csv to be read
- Return
    - Processed lexicon: (*df*)
- Write
    - None

### `export_data()`
Writes the given DataFrame to storage.
- Params
    - `sample_name` (*str*): the name of the sample; used to construct filename
    - `data` (*df*): the DataFrame to export
    - `extension` (*str*): the extension to save the df as. optional; defaults to `.CSV`
- Return
    - None
- Write
    - `data/train.{sample_name}{i}{extension}` for index `i`

### `export_df()`
A more generalized version of `export_data()`. Doesn't prepend "train" to the filename and allows different filepaths.
- Params
    - `data` (*df*): the DataFrame to export
    - `sample` (*str*): the type of sample + part of filename; can be blank
    - `i` (*int*): the index + part of filename; can be blank
    - `path` (*str*): the path to save the file; leave blank to save to CWD
    - `prefix` (*str*): the prefix of the filename (e.g. "topic", "report", etc.); can be blank
    - `index` (*bool*): if **TRUE**, write row names; default **TRUE**
- Return
    - None
- Write
    - `{path}/{prefix}.{sample}{i}.CSV`


## kaggle_preprocessing.py
This file reformats and cleans the data from `kaggle_build.py` into something the SVM can use.

### `read_data()`
Reads given DataFrame line-by-line. Some comments have tabs or commas, and that can cause issues depending on the
file delimiter. Removes entries with missing values (there's only 1 in `train.target+comments.TSV` without a score)
- Params
    - `dataset` (*str*): filename of dataset to import
    - `verbose` (*verbose*): toggles print statements; default **TRUE**
- Return
    - Clean delimited data: (*df*)
- Write
    - None
    
### `format_kaggle_toxicity()`
Reads `kaggle_toxic` training file and cleans it up + applies the correct header names.
- Params
    _ None
- Return
    - None
- Write
    - `data/src_new/kaggle-toxic_train-clean.CSV`

### `sample_data()`
Given a DataFrame, shuffle it and cut it down to the given size.
- Params
    - `data` (*df*): data to sample
    - `size` (*int*): sample size
- Return
    - Sampled data: (*df*)
- Write
    - None

### `boost_data()`
Given data, return only rows containing predefined abusive words. Or, if given a wordbank, return rows containing any
of those words instead.
- Params
    - `data` (*df*): DataFrame to boost
    - `data_name` (*str*): filename for print statements; ignored if `verbose=FALSE`
    - `verbose` (*bool*): controls verbosity; default **TRUE**
    - `manual_boost` (*[str]*, or *None*): user-defined wordbank to boost on; default *None*
- Return
    - Boosted data: (*df*)
- Write
    - None


## kaggle_train.py
This file trains `n` SVMs for all three sample types, with `n` being the `repeats` flag.

### `fit_data()`
This is where the magic happens. Fits CountVectorizer, trains SVM, and prints + exports results per dataset.
- Params
    - `rebuild` (*bool*): if TRUE, rebuild + rewrite the following datasets:
    - `samples` (*[str]*): three modes: "random", "boosted", or "all"
    - `analyzer` (*str*): either "word" or "char". for CountVectorizer
    - `ngram_range` (*(int,int)*): tuple containing lower and upper ngram bounds for CountVectorizer
    - `manual_boost` (*[str]*): use given list of strings for filtering instead of built-in wordbanks. Or pass `None`
    - `repeats` (*int*): controls the number of datasets built per sample type (if `rebuild` is **TRUE**)
    - `verbose` (*boolean*): toggles print statements
    - `sample_size` (*int*): size of sampled datasets. If set too high, the smaller size will be used
    - `calc_pct` (*bool*): if **TRUE**, calculate percentage of explicitly abusive and implicitly abusive words in each sample
    - `decimals` (*int*): number of decimals to round percentages to
- Return
    - None
- Write
    - `output/pred/pred.{sample_type}{i}` for index `i` and string `sample_type`, both defined in-function
    - `output/stats/percent_abusive/percent.{sample_type}{i}` if `calc_pct` is **TRUE**
    - `output/report/report.{sample_type}{i}`

### `import_data()`
Helper function that queues datasets to be trained *per sample*. It reads `n` sets for the given `sample_type`
- Params
    - `sample_type` (*str*): part of filename, used for reading it into memory
    - `n` (*int*): number of files per sample
- Return
    - List of DataFrames: (*[df]*)
- Write
    - None
    
### `pred_helper()`
Helper function that checks for previously-computed `y_pred`. If it exists, print it; else, compute it.
- Params
    - `x` (*df*): data to predict
    - `y` (*df*): class vector
    - `clf` (*sklearn.pipeline.Pipeline*): CountVectorizer and SVM models
    - `k` (*int*): number of folds to be used in cross-validation
    - `sample_type` (*str*): name of sample type; used for filename checks + exports
    - `i` (*int*): index; used for filename checks + exports
    - `verbose` (*bool*): used to control verbosity of import / fit steps
- Return
    - None
- Write
    - `output/pred/pred.{sample_type}{i}.CSV` if y_pred doesn't already exist for sample type and index `i`

### `pct_helper()`
Helper function that checks for previously-computed abusive-content percentages. If it exists, print it; else, compute it.
- Params
    - `data` (*df*): data to compute percentages for
    - `sample_type` (*str*): name of sample type; used for filename checks + exports
    - `i` (*int*): index; used for filename checks + exports
    - `verbose` (*bool*): used to control verbosity of import / fit steps
- Return
    - None
- Write
    - `output/stats/percent_abusive/percent.{sample_type}{i}.CSV` if percentage doesn't already exist for sample type and index `i`

### `main()`
Wrapper, called from real main. Protects inner-scope variables in `fit_data()`
- Params
    - None
- Return
    - None
- Write
    - None

## kaggle_postprocessing.py
If postprocessing wasn't already a word, it is now. This contains helper functions that work with data that has already been trained or processed.

### `calc_pct_abusive()`
This computes how much of `data` is considered abusive. Uses all three lexicons: *manual*, *Wiegand Base*, and 
*Wiegand Extended*. Returns a DataFrame with a column of lexicon names and calculated percentages.
- Params
    - `data` (*df*): DataFrame to calculate abusive contents of
- Return
    - DataFrame of results: (*df*)
- Write
    - None
    
TODO: Utilize the `multiprocessing` library for Python for parallel boosts. I got it to work but it hung up when joining jobs due to the Queue object.

### `calc_oov()`
This simulates 5-fold cross validation on the sampled datasets (9 total), then calculates the percentage of out-of-vocabulary
words per fold.
- Params
    - `k` (*int*): number of folds to use for cross-validation
    - `verbose` (*bool*): toggle verbosity of function
- Return
    - None
- Write
    - `stats/oov/oov.{sample_type}{i}` for index `i` and string `sample_type`, both defined in-function 
    
#### `manual_kfold()`
Given a DataFrame, split into `k` folds and return list of train + test splits
- Params
    - `data` (*df*): the data to split
    - `k` (*int*): number of folds to use for cross-validation
    - `state`(*int*): controls the random state of `sklearn.model_selection.KFold`
- Return
    - List of lists of DataFrames: (*[[train1, test1], [train2, test2]...]*)
        - each train/test pair is for one fold
- Write
    - None
    
#### `get_usage_sets()`
Given a DataFrame and lexicon, return two sets: the words in both the df and lexicon (used words), and the words in the lexicon
but not the df (unused words)>
- Params
    - `df` (*df*): the DataFrame to 
    - `lex` (*[str]*)
- Return
    - Set of used words (*{}*)
    - Set of unused words (*{}*)
- Write
    - None

##### `get_words_list()`
Given a DataFrame of the correct format, create a set of words used in its `comment_text` feature.
- Params
    - `data` (*df*): the DataFrame to process
- Return
    - Set of strings (*{str}*)
- Write
    - None

# Afterword
Thanks for reading, and happy training!
