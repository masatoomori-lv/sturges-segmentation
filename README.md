# Sturges Segmentation

This repository provides a method to find threshold values for segmenting a target variable represented in a binary `[0, 1]` format, using Sturges' formula to segment continuous explanatory variables.
It calculates the optimal threshold for each combination of two explanatory variables.

TODO: Add support for categorical variables ([Issue #3](https://github.com/masatoomori-lv/sturges-segmentation/issues/3)).

## Data

Place a CSV file containing the explanatory variables and the target variable in `./data/input`.

Also, place a metadata file with the same file name in `./data/` input and describe it as follows.
If a metadata file does not exist, all explanatory variables are treated as continuous variables (TBD: [Issue #3](https://github.com/masatoomori-lv/sturges-segmentation/issues/3)), and the column representing the target will be the first column.
Columns not listed under `continuous` or `categorical` will not be used.

```json
{
  "target": "target",
  "continuous": ["continuous1", "continuous2"],
  "categorical": ["categorical1", "categorical2"]
}
```

### Example Data

The example data used in the Sturges Segmentation project is the wine dataset from scikit-learn.
This dataset is utilized to demonstrate the application of Sturges' formula, adjusting the target variable to fit a binary classification schema.

#### Data Description

The wine dataset comprises several chemical constituents found in wines grown in the same region in Italy but derived from three different cultivars.
The dataset features include quantities such as alcohol content, malic acid, ash, and other attributes relevant to wine analysis.

#### Modifications for Binary Classification

The original target variable in the wine dataset categorizes the wines into three different cultivars.
For the purposes of this project, the target variable is transformed into a binary format:

- The most frequent category (mode) is labeled as 1.
- All other categories are labeled as 0.

This transformation simplifies the classification task and allows the use of binary logistic regression models or other binary classifiers for predictive modeling.

#### Data Preparation and Storage

1. Downloading and Transformation: The script `download_example.py` is used to download the wine dataset and apply the binary classification transformation as described above. The script modifies the target column based on the most frequent value and adjusts other data fields as necessary.
1. File Naming and Location:
   - The transformed data is saved as `example_data.csv` by default.
   - Users can specify a different file name by using the `--file_name` command-line argument when running the script.
   - The dataset is stored in the `./data/input` directory. If the `INPUT_DATA_DIR` environment variable is set, the directory specified by this variable will be used instead.

#### Running the Script

To download and prepare the example data, navigate to the script's directory and execute:

```bash
cd ./src/data
python download_example.py --file_name example_data.csv
```

This command will fetch the dataset, apply the necessary transformations, and save it to the designated location.

## Usage
