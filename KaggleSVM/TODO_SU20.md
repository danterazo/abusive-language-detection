# TODO, after SPR break III
- [] URGENT: finish hate speech list
    - look over manual tags again
    - 4 classes: not hate, mildly abusive, very abusive, ?
      - 0, 1, 2, n
    - abusive = you say if you want to hurt someone
- [] Figure out differences between R-imported and Python data
    - Unix `diff`. Save train from R, use `diff`, then compare two
- [] Ask Sandra to change delimiter (?)
- [] count how often people agree
    - consider collapsing mildly + very abusive
    - look at ones with low agreement
    - consider agreement rate, that determines how far we go
    - might have to do one last round
    - Wiegand was automatic, hence uncommon false flags
- [] add other TODO items from unsorted list


# 4-23 TODO
- email whenever you upload stuff to Box
- UPLOAD hatespeech lexicon
- [] compare datasets: 3-4 abusive, take
    - 2 abusive, remove and have people decide later
- New lexicon files: `lex.tox`
- up wordbank to get 10000 posts
- [x] write a script to get 10000 posts randomly from training set
    - run ML on 10000 random and 10000 filtered to compare
    - [x] save 10000 posts to file + share. don't get 10000 random each time (one fixed random dataset)
- [] share python scripts/files on Box, send email to both
- Let Sandra know if you're busy over the summer
    - Could postpone to FA20

# 4-30, 5-15 TODO
- `lex.general` most common words + appearances
- [] might be an issue with my code
- should be 176048 trump examples. if so, just use those
    - `grep -i -c "trump" train.target+comments.tsv`
    - 10K randomly sampled trump examples (?)
- [] workshop, july 23, see email for details
    
# For 2020-05-21
- after extracting 176K from Trump, randomly choose `20000`
- ~~wordbank: focus on most controversial topics~~
    - ~~trim to only good political ones~~
    - ~~then rebuild random dataset~~
    - ~~up random dataset to `20000` filtered tweets~~
- so three datasets:
    - two subconditions for **Biased Sampling**
        - [x] boosted: 20000 filtered on "trump"
            - filter entire `train.target+comments.tsv`
            - select 20000 random
            - more for fun instead of results
        - [x] boosted: 20000 filtered on wordbank
            - filter entire `train.target+comments.tsv`
            - select 20000 random
            - Closer to Wiegand's boosted random sampling
    - Kaggle was already **random-boosted sampling**
        - [x] random: 20000 randomly picked from unfiltered dataset
            - save to `train.random.csv`
- replicating paper: three datasets with different sampling strategies 
    - minimal pairs, different in exactly one thing
- after datasets
    - [] do GridSearch on random 20K dataset, print parameters
        - `https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html`
        - use best GS settings to do 5CV
    - [] convert to 5-fold cross validation
        - potential point of contention by reviewer panel
        1. Run 5CV for all of them (don't split into train/test)
        2. sklearn takes care of train/test
        3. `dev` doesn't matter anymore, reimplement
            - also, `dev` is used until last step. then `test`


# 2020-05-22
- interpreting results
    - look at `macro avg`, takes into account more than just avg
        - weights offensive examples (more sparse) higher than nonoffensive
- big idea: **gold standard**
- sample everything 3 times, then average results
- fixed datasets: shuffle data, then save `.tsv`
    - rerunning experiments `.tsv`
    - e.g. `Trump1`, `Trump2`, `Trump3`
        - all 20000 randomly picked from big dataset
        - move data import / filtering code to kaggle_build?
    - don't worry about splits, bc 5CV
    - [x] Random x3
    - [x] Topic x3
    - [x] Wordbank x3
- after all that:
    - [] Compare 3 manually annotated lexicons, minus Brooklyn's
    - [] save to another `.csv`
    - [] filter on lexicon to see how many explicitly abusive comments there are based on OUR lexicon
    - [] next step: sample 20000, but only explicit or implicit


# 2020-05-29
- Ask Sandra to look at final lexicon. We need to decide which words to use
    - cross-checking original author offensive words with ours
- ~~new report row -> average of all columns~~
- [x] save output using `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict`
    - [x] then send that off to `classification_report()`
- don't hear back from Sandra in three days? message again
- script README w/ config, setup, etc.
