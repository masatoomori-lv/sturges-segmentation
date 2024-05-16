#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'dotenv')
get_ipython().run_line_magic('dotenv', '')


# In[ ]:


import os
import math
from typing import List, Tuple
from logging import getLogger, StreamHandler, DEBUG

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import pygwalker as pyg


# In[ ]:


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


# ## environment variables

# In[ ]:


# Parameters
MIN_COMP_RATIO = 0.05
NICE_ROUND = False
OUTPUT_FORMAT = "csv"   # "csv" or "xlsx"


# In[ ]:


# Internal constants
RANDOM_STATE = 42
DEFAULT_INPUT_FILE = 'example_data.csv'


# In[ ]:


# Environment variables
INPUT_DATA_DIR = os.environ.get('INPUT_DATA_DIR')
OUTPUT_DATA_DIR = os.environ.get('OUTPUT_DATA_DIR')
assert INPUT_DATA_DIR is not None
assert OUTPUT_DATA_DIR is not None


# ## functions

# In[ ]:


def load_data(file_name: str=DEFAULT_INPUT_FILE):
    f = os.path.join(INPUT_DATA_DIR, file_name)
    return pd.read_csv(f), file_name


# In[ ]:


def round_nice(numbers: List[float], data_min: float, data_max: float, n_digits: int=1) -> List[float]:
    # TODO: use n_digits
    rounded_numbers = list()
    for number in numbers:
        exponent = math.floor(math.log10(number))
        base = 10 ** exponent
        factor = number / base

        # Determine if it is close to 1, 2, or 5
        if factor < 1.5:
            rounded_numbers.append(1 * base)
        elif factor < 3.5:
            rounded_numbers.append(2 * base)
        else:
            rounded_numbers.append(5 * base)

    rounded_numbers = sorted(list(set(rounded_numbers)))

    # if min value in original numbers is less than min of rounded numbers, then add a smaller number
    if data_min < min(rounded_numbers):
        rounded_numbers.insert(0, data_min)
    # if max value in original numbers is greater than max of rounded numbers, then add a larger number
    if data_max > max(rounded_numbers):
        rounded_numbers.append(data_max)

    return rounded_numbers


# In[ ]:


def calculate_thresholds(df: pd.DataFrame, col: str, nice_round: bool) -> List[float]:
    n = df[col].count()
    k = int(1 + math.log2(n))  # Sturges' formula

    unique_values = df[col].unique().tolist()
    if len(unique_values) <= 1:
        return list()

    if len(unique_values) <= k:
        cut_points = sorted(unique_values)
    else:
        bins = pd.qcut(df[col], q=k, duplicates='drop')
        cut_points = [df[col].min()] + [bins.cat.categories[i].right for i in range(len(bins.cat.categories))]

    if nice_round:
        # Round to the nearest 1, 2, or 5 multiples of one significant digit
        cut_points = round_nice(cut_points, data_min=min(df[col]), data_max=max(df[col]))

    return cut_points


# In[ ]:


def bin_records(df: pd.DataFrame, col: str, nice_round: bool) -> pd.DataFrame:
    """
    Merge adjacent records with adjacent values in col and the same predicted result
    """
    cut_points = calculate_thresholds(df, col, nice_round=nice_round)

    # Create a new column with the bin number
    df[f'bin_{col}'] = pd.cut(df[col], bins=cut_points, labels=None, include_lowest=True)
    df[f'bin_{col}_lower'] = df[f'bin_{col}'].apply(lambda x: x.left)
    df[f'bin_{col}_upper'] = df[f'bin_{col}'].apply(lambda x: x.right)
    df[f'bin_{col}_str'] = df[f'bin_{col}'].apply(lambda x: f'{x.left} < {col} <= {x.right}')

    return df


# In[ ]:


def aggregate_records(df: pd.DataFrame, target_col: str, pred_col: str, feature_cols: List[str]) -> pd.DataFrame:
    feature_1 = feature_cols[0]
    feature_2 = feature_cols[1]
    feature_1_range_col = 'feature_1_range'
    feature_2_range_col = 'feature_2_range'
    feature_1_lower_col = f'bin_{feature_1}_lower'
    feature_1_upper_col = f'bin_{feature_1}_upper'
    feature_2_lower_col = f'bin_{feature_2}_lower'
    feature_2_upper_col = f'bin_{feature_2}_upper'
    leaf_node_col = 'leaf_node'
    df['n_samples'] = 1

    df = df.groupby([leaf_node_col]).agg({
        feature_1_lower_col: ['min'],
        feature_1_upper_col: ['max'],
        feature_2_lower_col: ['min'],
        feature_2_upper_col: ['max'],
        feature_1: ['mean'],
        feature_2: ['mean'],
        target_col: ['mean'],
        pred_col: ['mean'],
        'n_samples': ['sum']
    })
    df.columns = [col[0] for col in df.columns.values]
    df.reset_index(inplace=True)
    df[feature_1_range_col] = df.apply(lambda x: f'{x[feature_1_lower_col]} < {feature_1} <= {x[feature_1_upper_col]}', axis=1)
    df[feature_2_range_col] = df.apply(lambda x: f'{x[feature_2_lower_col]} < {feature_2} <= {x[feature_2_upper_col]}', axis=1)

    df = df.sort_values([feature_1_range_col, feature_2_range_col]).reset_index(drop=True)

    return df


# In[ ]:


def format_table(df: pd.DataFrame, target_col: str, pred_col: str, feature_cols: List[str], base_value: float) -> pd.DataFrame:
    df.rename(columns={
        feature_cols[0]: 'feature_1_mean',
        feature_cols[1]: 'feature_2_mean',
        'bin_{}_lower'.format(feature_cols[0]): 'feature_1_lower',
        'bin_{}_upper'.format(feature_cols[0]): 'feature_1_upper',
        'bin_{}_lower'.format(feature_cols[1]): 'feature_2_lower',
        'bin_{}_upper'.format(feature_cols[1]): 'feature_2_upper',
        # 'bin_{}_str'.format(feature_cols[0]): 'feature_1_range',
        # 'bin_{}_str'.format(feature_cols[1]): 'feature_2_range',
    }, inplace=True)
    df['feature_1'] = feature_cols[0]
    df['feature_2'] = feature_cols[1]
    df['proportion'] = df['n_samples'] / df['n_samples'].sum()
    df['base_value'] = base_value
    df['odds'] = df[target_col] / df['base_value']
    df = df[[
        'feature_1', 'feature_2',
        'feature_1_range', 'feature_2_range',
        target_col, pred_col,       # average values of target and prediction
        'n_samples', 'proportion',  # number of samples and proportion to total
        'base_value', 'odds',       # overall average of target and odds to base ratio
        'feature_1_lower', 'feature_1_mean', 'feature_1_upper',
        'feature_2_lower', 'feature_2_mean', 'feature_2_upper',
    ]]
    return df.copy()


# ## main

# In[ ]:


# main
df, file_name = load_data()


# In[ ]:


# sns.pairplot(df)


# In[ ]:


# put index to Details field
# pyg.walk(df.reset_index())


# In[ ]:


target_col = df.columns[0]
pred_col = f'{target_col}_pred'
feature_cols = df.drop(target_col, axis=1).columns.tolist()
base_value = df[target_col].mean()
min_samples = math.ceil(len(df) * MIN_COMP_RATIO)

# make pairs of feature columns
feature_col_pairs = [[feature_cols[i], feature_cols[j]] for i in range(len(feature_cols)) for j in range(i+1, len(feature_cols))]


# In[ ]:


df_master = pd.DataFrame()
for feature_col_pair in feature_col_pairs:
    # to align with sturges segmentation, classify records based on midpoints of bins
    df_x = df[feature_col_pair].copy()
    feature_cols = list()
    for feature_col in feature_col_pair:
        df_x = bin_records(df_x, feature_col, nice_round=NICE_ROUND)
        df_x[f'bin_{feature_col}_midpoint'] = df_x[[f'bin_{feature_col}_lower', f'bin_{feature_col}_upper']].mean(axis=1)
        feature_cols.append(f'bin_{feature_col}_midpoint')

    X = df_x[feature_cols]
    y = df[target_col]

    model = DecisionTreeClassifier(min_samples_leaf=min_samples, min_impurity_decrease=0, random_state=RANDOM_STATE)
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    leaf_nodes = model.apply(X)

    df_pred = df_x.copy()
    df_pred[target_col] = y
    df_pred[pred_col] = y_pred[:, 1]
    df_pred['leaf_node'] = leaf_nodes

    # plot_tree(model, feature_names=feature_col_pair, class_names=[f'not {target_col}', target_col], filled=True)

    # assessment
    accuracy = model.score(X, y)
    logger.debug(f'accuracy of {feature_col_pair}, {accuracy}')
    # fig = df_pred.plot.scatter(x=feature_col_pair[0], y=feature_col_pair[1], c=pred_col, colormap='viridis')
    # plt.show()

    df_pred = aggregate_records(df_pred, target_col, pred_col, feature_col_pair)
    df_pred = format_table(df_pred, target_col, pred_col, feature_col_pair, base_value)

    df_master = pd.concat([df_master, df_pred], axis=0).reset_index(drop=True)

df_master


# In[ ]:


# replace suffix of the output file
# check if OUTPUT_FORMAT is not in the file name
file_body, file_ext = os.path.splitext(file_name)
output_file = os.path.join(OUTPUT_DATA_DIR, file_body + '_segment.{}'.format(OUTPUT_FORMAT))
if OUTPUT_FORMAT == 'csv':
    df_master.to_csv(output_file, index=False)
elif OUTPUT_FORMAT == 'xlsx':
    df_master.to_excel(output_file, index=False)
else:
    raise ValueError(f'Invalid OUTPUT_FORMAT: {OUTPUT_FORMAT}')

