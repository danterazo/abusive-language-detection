# Introduction & Research Goals 
The goal of this project was to improve both implicit abusive language detection. **Python** was used for data preprocessing, dataset builds, and SVM training. **R** was used to verify dataset properties (e.g. length, headers, etc.). The paper was written in and compiled from **LaTeX**.

## Paper
The paper is a WIP but will include detailed research methods, results, citations, and more.

See `Paper` for uncompiled LaTeX and assets. The final paper will be in the root directory.

The plan is to submit this paper to the [_Fourth Workshop on Online Abuse and Harms_](https://www.aclweb.org/portal/content/fourth-workshop-online-abuse-and-harms) (WOAH), co-located with the 2020 conference on _Empirical Methods in Natural Language Processing_ (EMNLP 2020).

# Code
This repository contains all of the resources you will need to replicate results. Boosting data takes an absurd amount of time depending on the lexicon used, so I recommend using a computer with at least **4c/4t** and **16GB** of memory. Again, this is a recommendation and not a requirement.

## Running the script
1. Install dependencies
    - **Pip**: `pip3 install -r requirements.txt`
    - **Conda**: `conda install --file requirements.txt`
      - Select the desired conda `env` before installing

2. Clone
    - `git clone https://github.com/danterazo/abusive-language-detection.git`

3. Configure
    - See the main in `kaggle_train.py`
        | Variable     | Data Type    | Default Value | Possible Values                                      | Purpose                                                      |
        | ------------ | ------------ | ------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
        | `samples`      | *str*        | **"all"**     | "random", "boosted_topic", "boosted_wordbank", "all" | Lets the user choose which sample types to train on          |
        | `analyzer`     | *str*        | **"word"**    | "char", "word"                                       | Toggle n-gram analyzer                                 |
        | `ngram_range`  | *(int, int)* | **(1,3)**     | {i &#124; i&isin; Z<sup>+</sup>}                     | Lower and upper n-gram boundaries                            |
        | `manual_boost` | *[str]*      | **["trump"]** | any array of strings whose length, **None**          | If not **None**, override predefined wordbanks when boosting |
        | `rebuild`      | *bool*       | **False**     | True, False                                          | If **True**, resample + rebuild training data. Computationally expensive |
        | `repeats`      | *int*        | **3**         | {i &#124; i &isin; Z<sup>+</sup>, i > 0}             | Set the number of each sample type to build and train. Ignored if `rebuild` is **False**. |
        | `sample_size`  | *int*        | **20000**     | {i &#124; i &isin; Z<sup>+</sup>, i > 0}             | Set the size of each dataset when building. If any set has <2000 examples, the others will be trimmed to match it. Ignored if `rebuild` is **False**. |
        | `verbose`      | *bool*       | **True**      | True, False                                          | Controls verbose print statements. Passed to other functions like a react prop. |
        | `calc_pct`     | *bool*       | **True**      | True, False                                          | If **True**, calculate the percentage of abusive words in each sample. Uses *manual*, *Wiegand Base*, and *Wiegand Extended* lexicons. Very computationally expensive. |

4. Train
    - Once you've configured the script, simply run `kaggle_train.py`. No user input is required.
    - `python3 kaggle_train.py`

5. Wait patiently for results
    - Percentage calculation (see `calc_pct` above) is multithreaded for mild performance improvements.
    - **Rebuilt datasets** can be found in `data\`
    - **Class predictions** can be found in `output\pred\`
    - **Classification reports** can be found in `output\report\`



## Nomenclature
RDS



## Files
In order of execution:

### `kaggle_build.py`
This file builds, imports, and exports data stored locally.
#### `build_train()`

### `kaggle_preprocessing.py`
This file reformats and cleans the data from `kaggle_build.py` into something the SVM can use.

### `kaggle_train.py`
This file trains `n` SVMs for all three sample types, with `n` being the `repeats` flag.

### `kaggle_postprocessing.py`




