"""
Download wine data as example data for testing from scikit-learn to the input data directory.
Default file name is 'example_data.csv'
To change the file name, input the file name as an argument

Downloaded data is converted to binary classification.
Most frequent class is assigned as positive class and the rest as negative class.
"""

import os
import argparse

from sklearn.datasets import load_wine
import pandas as pd

DOWNLOAD_DIR = os.environ.get('INPUT_DATA_DIR')
EXAMPLE_DATA_FILE = 'example_data.csv'


def download_data() -> pd.DataFrame:
    """Fetches the wine dataset from scikit-learn and returns it as a pandas DataFrame."""
    df = load_wine(as_frame=True).frame
    # move target column to the beginning
    df = df[['target'] + [col for col in df.columns if col != 'target']]
    return df


def convert_to_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    # Identify the most frequent class
    mode_value = df['target'].mode()[0]
    # Convert the target to binary: 1 if the most frequent class, 0 otherwise
    df['target'] = df['target'].apply(lambda x: 1 if x == mode_value else 0)
    return df


def save_data(df: pd.DataFrame, file_name: str):
    """Saves the DataFrame to a CSV file in the specified directory.
    Raises an error if the directory does not exist.
    """
    # Check if the directory exists, raise an error if not
    if not os.path.exists(DOWNLOAD_DIR):
        raise FileNotFoundError(f"Directory '{DOWNLOAD_DIR}' does not exist. Please create it and try again.")

    # Construct the full file path
    file_path = os.path.join(DOWNLOAD_DIR, file_name)

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def main():
    """Main function to handle command-line arguments and orchestrate downloading and saving data."""
    parser = argparse.ArgumentParser(description="Download wine dataset and save it to a file")
    parser.add_argument("--file_name", default=EXAMPLE_DATA_FILE, help="Name of the file to save the data. Default is 'example_data.csv'.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Download data
    df = download_data()

    # Convert to binary classification
    df = convert_to_binary_classification(df)

    # Save data to the specified or default file name
    save_data(df, args.file_name)

if __name__ == "__main__":
    main()
