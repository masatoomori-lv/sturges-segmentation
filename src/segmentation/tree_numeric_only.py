import os
import math
import argparse
from typing import List, Tuple
from logging import getLogger, StreamHandler, DEBUG

import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

# Internal constants
RANDOM_STATE = 42
DEFAULT_INPUT_FILE = 'example_data.csv'
PRED_COL_SUFFIX = '_pred'

# Environment variables
INPUT_DATA_DIR = os.environ.get('INPUT_DATA_DIR')
OUTPUT_DATA_DIR = os.environ.get('OUTPUT_DATA_DIR')
assert INPUT_DATA_DIR is not None
assert OUTPUT_DATA_DIR is not None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sturges Segmentation")
    parser.add_argument('--min_comp_ratio', type=float, default=0.05, help="Minimum composition ratio")
    parser.add_argument('--nice_round', action='store_true', help="Enable nice rounding")
    parser.add_argument('--output_format', type=str, default="csv", choices=["csv", "xlsx"], help="Output format")
    return parser.parse_args()


def load_data(file_name: str=DEFAULT_INPUT_FILE):
    f = os.path.join(INPUT_DATA_DIR, file_name)
    return pl.scan_csv(f), file_name


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


def bin_records(df: pd.DataFrame, col: str, nice_round: bool) -> pd.DataFrame:
    cut_points = calculate_thresholds(df, col, nice_round=nice_round)

    # Create a new column with the bin number
    df[f'bin_{col}'] = pd.cut(df[col], bins=cut_points, labels=None, include_lowest=True)
    df[f'bin_{col}_lower'] = df[f'bin_{col}'].apply(lambda x: x.left)
    df[f'bin_{col}_upper'] = df[f'bin_{col}'].apply(lambda x: x.right)
    df[f'bin_{col}_str'] = df[f'bin_{col}'].apply(lambda x: f'{x.left} < {col} <= {x.right}')

    return df


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


def format_table(df: pd.DataFrame, target_col: str, pred_col: str, feature_cols: List[str], base_value: float) -> pd.DataFrame:
    df.rename(columns={
        feature_cols[0]: 'feature_1_mean',
        feature_cols[1]: 'feature_2_mean',
        'bin_{}_lower'.format(feature_cols[0]): 'feature_1_lower',
        'bin_{}_upper'.format(feature_cols[0]): 'feature_1_upper',
        'bin_{}_lower'.format(feature_cols[1]): 'feature_2_lower',
        'bin_{}_upper'.format(feature_cols[1]): 'feature_2_upper',
    }, inplace=True)
    df['feature_1'] = feature_cols[0]
    df['feature_2'] = feature_cols[1]
    df['proportion'] = df['n_samples'] / df['n_samples'].sum()
    df['base_value'] = base_value
    df['odds'] = df[target_col] / base_value
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


def main():
    args = parse_arguments()

    df, file_name = load_data()

    record_count = df.select(pl.count()).collect().item()
    logger.debug(f'Number of records: {record_count}')

    target_col = df.columns[0]
    pred_col = target_col + PRED_COL_SUFFIX
    feature_cols = [col for col in df.columns if col != target_col]
    base_value = df.select(target_col).mean().collect().item()
    min_samples = math.ceil(record_count * args.min_comp_ratio)
    feature_col_pairs = [[feature_cols[i], feature_cols[j]] for i in range(len(feature_cols)) for j in range(i+1, len(feature_cols))]

    df_master = pd.DataFrame()
    for feature_col_pair in feature_col_pairs:
        # to align with sturges segmentation, classify records based on midpoints of bins
        df_x = df.select(feature_col_pair).collect().to_pandas()
        feature_cols = list()
        for feature_col in feature_col_pair:
            df_x = bin_records(df_x, feature_col, nice_round=args.nice_round)
            df_x[f'bin_{feature_col}_midpoint'] = df_x[[f'bin_{feature_col}_lower', f'bin_{feature_col}_upper']].mean(axis=1)
            feature_cols.append(f'bin_{feature_col}_midpoint')

        X = df_x[feature_cols]
        y = df.select(target_col).collect().to_numpy().flatten()

        model = DecisionTreeClassifier(min_samples_leaf=min_samples, min_impurity_decrease=0, random_state=RANDOM_STATE)
        model.fit(X, y)
        y_pred = model.predict_proba(X)
        leaf_nodes = model.apply(X)

        df_pred = df_x.copy()
        df_pred[target_col] = y
        df_pred[pred_col] = y_pred[:, 1]
        df_pred['leaf_node'] = leaf_nodes

        # assessment
        accuracy = model.score(X, y)
        logger.debug(f'accuracy of {feature_col_pair}, {accuracy}')

        df_pred = aggregate_records(df_pred, target_col, pred_col, feature_col_pair)
        df_pred = format_table(df_pred, target_col, pred_col, feature_col_pair, base_value)

        df_master = pd.concat([df_master, df_pred], axis=0).reset_index(drop=True)

    file_body, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(OUTPUT_DATA_DIR, file_body + '_segment.{}'.format(args.output_format))
    if args.output_format == 'csv':
        df_master.to_csv(output_file, index=False)
    elif args.output_format == 'xlsx':
        df_master.to_excel(output_file, index=False)
    else:
        raise ValueError(f'Invalid OUTPUT_FORMAT: {args.output_format}')


if __name__ == '__main__':
    main()
