# Sturges Segmentation

Sturges の公式を用いて、説明変数を分割し、`[0, 1]` の 2 直で表されるターゲット変数を分割するための閾値を求める。
説明変数は、連続直、カテゴリ変数のどちらでも対応可能。

## Data

`./data/input` に説明変数とターゲット変数を格納した csv ファイルを配置する。

同じファイル名の metadata ファイルを `./data/input` に配置し、以下のように記述する。
metadata ファイルが存在しない場合、全ての説明変数を連続変数として扱い（TBD: [Issue #3](https://github.com/masatoomori-lv/sturges-segmentation/issues/3)）、target を表すカラムは最初のカラムとする。

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

## Usage

オプションに `nice_round` を指定すると、閾値を有効数字 1 桁の 1, 2, 5 で丸める。

```bash
cd src
python run.py --nice_round
```

`./data/output` に、説明変数を分割するための閾値を格納した csv ファイルが出力される。
