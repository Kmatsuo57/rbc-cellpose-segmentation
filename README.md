# 赤血球の個体セグメンテーション（Cellpose）

赤血球を1細胞ずつ分離するためのCellpose-SAMパイプラインです。現在の運用対象はCellposeのみです。入力画像は変更せず、すべての生成物を `outputs/` に保存します。

## フォルダ構成

```text
data/raw/                 入力画像（読み取りのみ）
data/annotations/         手動正解マスクを作る場合の配置先
scripts/                  Cellpose推論・画像調査・アノテーション検証
rbc_segmentation/         共通I/O処理
outputs/                  Cellpose結果（Git管理外）
outputs/summary/          確認しやすい代表結果
```

## セットアップ

Python仮想環境を作り、プロジェクト直下で次を実行します。

```bash
python3 -m venv .venv-cellpose
.venv-cellpose/bin/python -m pip install -r requirements.txt
```

## 画像の確認

画像は `data/raw/` に置きます。現在の `0.jpg` と `10.jpg` はどちらも過酸化水素条件で、ラベルはそれぞれ `H2O2 0`、`H2O2 10` です。

```bash
.venv-cellpose/bin/python -m scripts.inspect_images
```

## Cellpose推論

現在の画像では `cpsam_v2`、RGB、0.5倍推論を基本条件とします。モデルのダウンロード先もプロジェクト内に固定します。

```bash
CELLPOSE_LOCAL_MODELS_PATH="$PWD/.cache/cellpose/models" \
  .venv-cellpose/bin/python -m scripts.predict_cellpose \
  --mode rgb --scale 0.5 --min-size 200 --output outputs/cellpose_run
```

特定画像だけ処理する場合は、たとえば次のように指定します。

```bash
CELLPOSE_LOCAL_MODELS_PATH="$PWD/.cache/cellpose/models" \
  .venv-cellpose/bin/python -m scripts.predict_cellpose \
  --filename 10.jpg --mode rgb --scale 0.5 --min-size 200 \
  --output outputs/cellpose_run
```

各画像について、元画像preview、正規化画像、instance label TIFF、輪郭overlay、個体計測CSV、設定JSONを出力します。代表結果と現時点の評価は `outputs/summary/README.md` を参照してください。

## 手動正解マスクを用意する場合

元画像と同じstemの整数ラベルTIFFを `data/annotations/labels/` に置き、対応画像を `data/annotations/images/` に置きます。背景は0、各細胞は1以上の重複しない連番IDにします。

```bash
.venv-cellpose/bin/python -m scripts.validate_annotations
```
