# CS 4740 Fall 2019 Project 2 -- metaphor detection with sequence labeling

## Data
In the folder Validation Results, we have all csv files by running two models model1 and model2 with different features on validation file val.csv.

## Prerequisites
Since we use nltk to train model2, you need to make sure the library has been installed.

## Experiment Files
These are files we used for doing experiments to choose best parameters and no need to run.

## HelperFunction Files and 
These are files we will use in other python files, no need to run.

## Python Files
For python files with prefix model1 or model2, choose a model and one feature we want, find the corresonidng file name and simply replace the path of test_file. Then run it, we can get the output csv file.


## Eval

python eval.py --pred sample_out.csv

(replace sample_out.csv with 'your prediction file name')
