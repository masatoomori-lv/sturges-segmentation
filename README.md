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

Output files are saved in `./data/output` for default settings.
If the `OUTPUT_DATA_DIR` environment variable is set, the directory specified by this variable will be used instead.

### Command Line Arguments

The script accepts the following command line arguments to control its behavior:

- `--min_comp_ratio`: Minimum composition ratio (default is 0.05).
- `--nice_round`: Enable nice rounding (default is False). This is a flag; if specified, it enables nice rounding.
- `--input_file`: Input file name (default is `example_data.csv`).
- `--output_format`: Output format, either `csv` or `xlsx` (default is `csv`).

### Running the Script

To run the script, navigate to the directory containing the script and execute:

```bash
cd ./src/segmentation
python tree_numeric_only.py --min_comp_ratio 0.1 --nice_round --output_format xlsx
```

This command will:

- Set the minimum compression ratio to 0.1.
- Enable nice rounding.
- Set the output format to xlsx.

### Output Format

The output file is saved in Excel format and contains the following columns:

| Column Name     | Description                                    |
| --------------- | ---------------------------------------------- |
| feature_1       | explanatory variable name 1                    |
| feature_2       | explanatory variable name 2                    |
| feature_1_range | rage of explanatory variable 1                 |
| feature_2_range | rage of explanatory variable 2                 |
| target          | mean value of target in this segment           |
| target_pred     | mean value of predicted target in this segment |
| n_samples       | number of samples in this segment              |
| proportion      | proportion to total sample size                |
| base_value      | mean value of target in the entire records     |
| odds            | mean value of target to base value             |
| feature_1_lower | lower range of explanatory variable 1          |
| feature_1_mean  | mean value of explanatory variable 1           |
| feature_1_upper | upper range of explanatory variable 1          |
| feature_2_lower | lower range of explanatory variable 2          |
| feature_2_mean  | mean value of explanatory variable 2           |
| feature_2_upper | upper range of explanatory variable 2          |
