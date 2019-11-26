# utility functions for loading data 
import sys
sys.path.insert(0,'/Users/neelguha/Dropbox/NeelResearch/fairness/code/tensorflow_constrained_optimization/')
import math
import random
import numpy as np
import pandas as pd
import warnings
from six.moves import xrange
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm

if 'fairness_data' in os.environ:
    DATA_DIR = os.environ['fairness_data']
else:
    raise Exception("Please set environment variable: fairness_data to point towards the data directory")

if 'results_dir' in os.environ:
    RESULTS_DIR = os.environ['results_dir']
else:
    raise Exception("Please set environment variable: results_dir to point towards the results directory")

def load_adult_data():
    ''' Loads adult dataset. 

        Returns: 
            train_df: training dataframe
            test_df: test dataframe
            feature_names: list of names of features
            protected_columns: list of columns corresponding to protected attributes
            label_column: name of column holding label 
    '''
    CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country'
    ]
    CONTINUOUS_COLUMNS = [
        'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
    ]
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket'
    ]
    LABEL_COLUMN = 'label'

    PROTECTED_COLUMNS = [
        'gender_Female', 'gender_Male', 'race_White', 'race_Black'
    ]

    data_dir = os.path.join(DATA_DIR, "adult")
    train_df_raw = pd.read_csv(os.path.join(data_dir, "adult.data"), names=COLUMNS, skipinitialspace=True)
    test_df_raw = pd.read_csv(os.path.join(data_dir, "adult.test"), names=COLUMNS, skipinitialspace=True, skiprows=1)

    train_df_raw[LABEL_COLUMN] = (train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    test_df_raw[LABEL_COLUMN] = (test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    # Preprocessing Features
    pd.options.mode.chained_assignment = None  # default='warn'

    # Functions for preprocessing categorical and continuous columns.
    def binarize_categorical_columns(input_train_df, input_test_df, categorical_columns=[]):

        def fix_columns(input_train_df, input_test_df):
            test_df_missing_cols = set(input_train_df.columns) - set(input_test_df.columns)
            for c in test_df_missing_cols:
                input_test_df[c] = 0
                train_df_missing_cols = set(input_test_df.columns) - set(input_train_df.columns)
            for c in train_df_missing_cols:
                input_train_df[c] = 0
                input_train_df = input_train_df[input_test_df.columns]
            return input_train_df, input_test_df

        # Binarize categorical columns.
        binarized_train_df = pd.get_dummies(input_train_df, columns=categorical_columns)
        binarized_test_df = pd.get_dummies(input_test_df, columns=categorical_columns)
        # Make sure the train and test dataframes have the same binarized columns.
        fixed_train_df, fixed_test_df = fix_columns(binarized_train_df, binarized_test_df)
        return fixed_train_df, fixed_test_df

    def bucketize_continuous_column(input_train_df, input_test_df, continuous_column_name, num_quantiles=None, bins=None):
        assert (num_quantiles is None or bins is None)
        if num_quantiles is not None:
            train_quantized, bins_quantized = pd.qcut(
              input_train_df[continuous_column_name],
              num_quantiles,
              retbins=True,
              labels=False)
            input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins_quantized, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins_quantized, labels=False)
        elif bins is not None:
            input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins, labels=False)

    # Filter out all columns except the ones specified.
    train_df = train_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]]
    test_df = test_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]]
    
    # Bucketize continuous columns.
    bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
    bucketize_continuous_column(train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
    bucketize_continuous_column(train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
    bucketize_continuous_column(train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
    bucketize_continuous_column(train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
    train_df, test_df = binarize_categorical_columns(train_df, test_df, categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
    feature_names = list(train_df.keys())
    feature_names.remove(LABEL_COLUMN)
    num_features = len(feature_names)

    return train_df, test_df, feature_names, LABEL_COLUMN


def get_ipums_income(small = False):
    ''' Returns ipums data. 

    ''' 
    
    data_dir = os.path.join(DATA_DIR, "ipums")
    if small: 
        train = pd.read_pickle(os.path.join(data_dir, 'small_train'))
        test = pd.read_pickle(os.path.join(data_dir, 'small_test'))
    else:
        train = pd.read_pickle(os.path.join(data_dir, 'train'))
        test = pd.read_pickle(os.path.join(data_dir, 'test'))

    label_column = 'INCOME_LABEL'
    
    # drop columns for other task 
    train.drop(columns=["HEALTH_LABEL"])
    test.drop(columns=["HEALTH_LABEL"])

    # determine valid protected columns 
    all_protected_prefixes = ['age', 'gender', 'race']
    protected_columns = []
    columns_to_keep = [label_column]
    feature_names = list(train.keys())
    feature_names.remove(label_column)
    return train, test, feature_names, label_column

def get_subsets(dataset):

    if dataset == 'adult-income':
        subsets = [
            ['gender_Female'], ['gender_Male'], ['race_White'], ['race_Black'],
            ['gender_Female', 'race_White'],
            ['gender_Female', 'race_Black'],
            ['gender_Male', 'race_White'],
            ['gender_Male', 'race_Black']
        ]
    else: 
        raise Exception("Uknown dataset: ", dataset)
    
    return subsets


def create_result_to_save(df, protected_columns, label_column):
    ''' In order to make future analysis of results easy, we want to save all information from a model's performance. 
        The most concise representation of this consists of: 
            - for each sample, the protected group membership, the prediction, and the true label 
            - the set of hyperparameters used in the experiment 
        
    Args: 
        df: a dataframe consisting of samples and their features
        protected_columns: a list of columns corresponding to 'protected attributes' 
        label_column: name of label column

    Returns: 
        out_df: a filtered df which only keeps protected attributes, predictions, and true labels 
    ''' 
    all_columns = list(df.keys())
    cols_to_keep = []
    for col in all_columns: 
        if col == label_column:
            cols_to_keep.append(col)
        elif col == 'predictions':
            cols_to_keep.append(col)
        else:
            for prefix in protected_columns:
                if prefix in col:
                    cols_to_keep.append(col)
                    break 
    df_filtered = df[cols_to_keep]
    df_filtered['predicted_class'] = df_filtered['predictions'] > 0.0
    return df_filtered


def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_protected_attributes(dataset_name, attributes, label = None, train = None, test = None):
    ''' Returns the list of protected attributes available for a dataset ''' 

    if dataset_name == 'adult-income':
        protected_attributes = []
        for attribute_name in attributes:
            
            # Education number 
            if 'education_num_' in attribute_name:
                protected_attributes.append(attribute_name)

            # Marital status 
            if 'marital_status_' in attribute_name:
                protected_attributes.append(attribute_name)

            # Race
            if 'race_' in attribute_name:
                protected_attributes.append(attribute_name)

            # Gender 
            if 'gender_' in attribute_name:
                protected_attributes.append(attribute_name)
            
            # Age 
            if 'age_' in attribute_name:
                protected_attributes.append(attribute_name)
                
        return protected_attributes
    elif dataset_name == 'ipums-small':
        protected_attributes = []
        for attribute_name in tqdm(attributes):
            
            is_age =  'age_' in attribute_name
            is_race = 'race_' in attribute_name
            is_gender = 'gender_' in attribute_name
            if not is_age and not is_race and not is_gender:
                continue 

            # check that attribute has positive samples in train and test 
            num_pos_train = sum(train[train[attribute_name] == 1][label].values)
            num_pos_test = sum(test[test[attribute_name] == 1][label].values)
            
            if num_pos_test > 10 and num_pos_train > 10:
                #print("{}: {} Pos. (Train) {} Pos (Test)".format(attribute_name, num_pos_train, num_pos_test))
                protected_attributes.append(attribute_name)
        return protected_attributes