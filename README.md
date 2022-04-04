# Direct Content Sensitivity Analysis and Prediction
## _Code for the paper_

This is the code for reproducing the results of the paper:

> _Analysis and classification of privacy-sensitive content in social media posts_ by Livio Bioglio and Ruggero G. Pensa (more details [here](#how-to-cite))

## Abstract

User-generated contents often contain private information, even when they are shared publicly on social media and on the web in general. Although many filtering and natural language approaches for automatically detecting obscenities or hate speech have been proposed, determining whether a shared post contains sensitive information is still an open issue. The problem has been addressed by assuming, for instance, that sensitive contents are published anonymously, on anonymous social media platforms or with more restrictive privacy settings, but these assumptions are far from being realistic, since post authors often underestimate or overlook their actual exposure to privacy risks. Hence, in this paper, we address the problem of content sensitivity analysis directly, by presenting and characterizing a new annotated corpus with around ten thousand posts, each one annotated as sensitive or non-sensitive by experts. We also present the results of several deep neural network models that outperform previous naive attempts of measuring sensitivity directly as well as state-of-the-art approaches based on the anonymity assumption.

## Details of the experiments

Our experiments use 3 datasets of textual posts, each one manually labeled with a single tag between:
- **sens**: the post contains a content that is sensitive for the privacy;
- **ns**: the post does not contain any content that is sensitive for the privacy.

The details of the three datasets are the following:

*Dataset*|*# posts*|*# sens*|*# ns*|*Avg # words*
---------|---------|--------|------|-------------
SENS2|8 765|3 336|5 429|15.11±12.58
SENS3|4 046|1 444|2 602|15.40±12.67
WH+TW|8 765|3 336|5 429|13.08±8.26

Datasets are available on request.

## Details of the code

The experiments have been performed using several python files and jupyter notebooks, all written in python 3.6, that makes use of the following modules:

- [Python 3.6+](https://www.python.org/) (tested on 3.6.7)
- [jupyter](http://jupyter.org/) (tested on 1.0.0)
- [numpy](https://numpy.org/) (tested on 1.18.1)
- [pandas](https://pandas.pydata.org/) (tested on 1.0.1)
- [matplotlib](http://matplotlib.org/) (tested on 3.1.3)
- [seaborn](https://seaborn.pydata.org/) (tested on 0.11.1)
- [nltk](https://www.nltk.org/) (tested on 3.4.5)
- [scikit-learn](https://scikit-learn.org/) (tested on 0.22.1)
- [tensorflow](https://www.tensorflow.org/) (tested on 2.3.0)
- [ktrain](https://github.com/amaiya/ktrain) (tested on 0.25.x)

Each file performs a precise task, as detailed in the following sub-sections.

### 1. Create datasets
The objective of this section is:
1. to create datasets _SENS2_ and _SENS3_ from the raw annotations;
2. to create dataset _WH+TW_ by composing a random subset of a large collection of [Whisper posts](https://github.com/Mainack/whisper-2014-2016-data-HT-2020) and a random subset of a large collection of [Twitter posts](https://dl.acm.org/doi/abs/10.1145/1871437.1871535);
3. to create Training, Validation and Test sets for each of these datasets.

The full code for this task is contained in the file `00 - Datasets Creation and Analysis.ipynb`.

### 2. Analysis of the lexical features

We categorize all words contained in each post into two different dictionaries:
1. **Psych**: provided by [LIWC](https://journals.sagepub.com/doi/abs/10.1177/0261927X09351676), a hierarchical linguistic lexicon that classifies words into meaningful psychological categories;
2. **Priv**: the [Privacy Dictionary](https://www.heinz.cmu.edu/~acquisti/SHB/p3227-gill.pdf), consisting in a dictionary of categories derived using prototype theory according to traditional theoretical approaches to privacy.

The entire pipeline of this task for all the datasets is contained into the file `01 - Dict class.ipynb`.

### 3. Build the Bag of Words (BoW) models

Bag-of-words (BoW) models consists of standard classifiers (Logistic Regression - LR; Support Vector Machine - SVM; Random Forests - RF) trained on _tfidf_ features extracted from text data after applying stemming and removing stopwords.

The entire pipeline of this task for all the datasets is contained into the file `02 - Bag of words.ipynb`.

### 4. Build the Recurrent Neural Networks (RNNs)

The datasets created in section 1 are employed for training four different RNN models: these models are trained to predict the tag (_sensible_ or _not sensible_) of a post.

This task is performed by the following files:
1. `experiment_01.py` trains the 4 RNNs for the _SENS2_ and _SENS3_ datasets;
2. `experiment_02.py` trains the 4 RNNs for the _WH+TW_ datasets;
3. `03 - [Sens] Analysis of RNNs.ipynb` analyzes the performances of the RNNs trained on the _SENS2_ and _SENS3_ datasets;
4. `03 - [WH+TW] Analysis of RNNs.ipynb` analyzes the performances of the RNNs trained on the _WH+TW_ datasets.

In addition, the file `experiment_00.py` trains several other RNNs for the _SENS2_ and _SENS3_ datasets: it is employed for detecting the best parameters for the datasets.

### 5. Build the BERT models

The datasets created in section 1 are employed for training a BERT model: these models are trained to predict the tag (_sensible_ or _not sensible_) of a post.

This task is performed by the following files:
1. `04 - [Sens] BERT.ipynb` trains the BERT models for the _SENS2_ and _SENS3_ datasets on the _colab_ platform;
2. `04 - [WH+TW] BERT.ipynb` trains the BERT models for the _WH+TW_ datasets on the _colab_ platform;
3. `05 - Analysis of BERT.ipynb` analyzes the performances of the BERT models trained on all the datasets.

### 6. Test _SENS2_ and _SENS3_ without biases

Since _SENS2_ and _SENS3_ derive from the same raw dataset of annotations, some of the posts in the test set of _SENS2_ are contained in the training set of _SENS3_, and the other way around: for this reason, there is a small bias in the analysis of the performances of a model trained on a dataset and testes on the other one. 

The code in `06 - [Sens] Clean test sets from training data.ipynb` tests the RNNs and the BERT models trained on _SENS2_ (_SENS3_) on the test set of _SENS3_ (_SENS2_) where all the posts in common with the training set of _SENS2_ (_SENS3_) have been removed.


## How to Cite

Please cite the following paper:

Bioglio, L., Pensa, R.G. **Analysis and classification of privacy-sensitive content in social media posts**. *EPJ Data Sci.* 11, 12 (2022). https://doi.org/10.1140/epjds/s13688-022-00324-y















