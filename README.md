# Sturges Segmentation

Sturges の公式を用いて、説明変数を分割し、`[0, 1]` の 2 直で表されるターゲット変数を分割するための閾値を求める。
説明変数は、連続直、カテゴリ変数のどちらでも対応可能。

## Data

`./data/input` に説明変数とターゲット変数を格納した csv ファイルを配置する。

同じファイル名の metadata ファイルを `./data/input` に配置し、以下のように記述する。

```json
{
  "target": "target",
  "continuous": ["continuous1", "continuous2"],
  "categorical": ["categorical1", "categorical2"]
}
```

## Usage

オプションに `nice_round` を指定すると、閾値を有効数字 1 桁の 1, 2, 5 で丸める。

```bash
cd src
python run.py --nice_round
```

`./data/output` に、説明変数を分割するための閾値を格納した csv ファイルが出力される。
