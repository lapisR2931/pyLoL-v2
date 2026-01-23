# pyLoL-v2: League of Legends 視界スコア分析による勝敗予測
##　はじめに

このレポジトリはpylolプロジェクトをフォークしたものです
引用元：https://github.com/league-of-legends-replay-extractor/pyLoL

##　このレポジトリで出来る事
- 2026/1/10時点でのLoLcliant、riotAPIを用いたroflファイル、match,timelieデータ収集フローの構築
- wardアイコンの検出（４種）
- ミニマップからward座標の推定、timelineデータをもとに設置・削除時刻の反映（精度80%前後）
- 視界スコアの重み計算と特徴量の作成
- 簡単な勝敗予測モデルを用いた効果測定

## 主な成果

| 指標 | 値 |
|------|-----|
| **勝敗予測精度（20分時点）** | 88.04% |
| **ベースラインからの改善** | +7.61 pt |
| **Riot公式visionScore比** | 約2倍の改善効果 |
| **wardマッチング精度** | 85%（ハンガリアン法） |

## 技術スタック

- **言語**: Python 3.7+
- **物体検出**: YOLOv8
- **機械学習**: scikit-learn, PyTorch
- **画像処理**: OpenCV, Pillow
- **API**: Riot Games API, LCU API
- **データ処理**: pandas, NumPy

## プロジェクト概要

### 背景

League of Legendsにおいて、視界（ward）の配置は勝敗に大きく影響する戦術要素です。しかし、Riot公式の`visionScore`は試合終了時のみ取得可能であり、途中経過での評価や配置位置の戦術的価値を反映していません。

本プロジェクトでは、リプレイ動画からwardの座標を抽出し、「どこに置いたか」ではなく「どれだけ良い位置に置いたか」を定量化する「戦術スコア」を開発しました。

### アプローチ

```
リプレイファイル(.rofl)
        ↓
    ミニマップ抽出（LCU API + 画面キャプチャ）
        ↓
    ward検出（YOLOv8）
        ↓
    座標抽出 + タイムラインマッチング（ハンガリアン法）
        ↓
    グリッド特徴量生成（32×32）
        ↓
    ヒートマップ学習 → 戦術スコア算出
        ↓
    勝敗予測モデルに統合
```

## 実験結果

### モデル性能比較（10分時点）

| モデル | Accuracy | AUC | 特徴量 |
|--------|----------|-----|--------|
| baseline（従来指標のみ） | 59.78% | 0.648 | 20個 |
| + Riot visionScore | 68.48% | 0.710 | +3個 |
| + wardグリッド統計 | 59.78% | 0.648 | +6個 |
| **+ 戦術スコア** | **77.17%** | **0.826** | +6個 |

### モデル性能比較（20分時点）

| モデル | Accuracy | AUC |
|--------|----------|-----|
| baseline | 80.43% | 0.901 |
| + Riot visionScore | 84.78% | 0.907 |
| **+ 戦術スコア** | **88.04%** | **0.945** |

### 主要な発見

1. **戦術スコアはRiot visionScoreの約2倍の予測改善効果**を示した
2. **序盤（10分時点）ほど視界スコアの効果が大きい** - 経済差が小さい時点では視界の質が重要
3. **単純なward座標では効果なし** - 位置の「戦術的価値」を評価することが必須

## ディレクトリ構成

```
pyLoL-v2/
├── autoLeague/                    # メインパッケージ
│   ├── dataset/                   # データ取得・生成
│   │   ├── generator.py           # matchId収集
│   │   ├── downloader.py          # リプレイダウンロード
│   │   ├── riotapi.py             # Riot API wrapper
│   │   └── ward_tracker.py        # タイムライン統合
│   ├── replays/                   # リプレイ操作
│   │   └── scraper.py             # ミニマップ抽出
│   ├── scoring/                   # 視界スコア
│   │   ├── grid_generator.py      # グリッド特徴量
│   │   ├── predictor.py           # 予測モデル
│   │   └── visualizer.py          # ヒートマップ可視化
│   └── prediction/                # 勝敗予測
│       ├── feature_extractor.py   # 特徴量抽出
│       ├── baseline_predictor.py  # 予測モデル
│       └── evaluator.py           # 評価・比較
├── notebooks/                     # 実行用Notebook
│   ├── 01_get_matchids.ipynb      # matchId収集
│   ├── 02_download_replays.ipynb  # リプレイDL
│   ├── 04_run_client.ipynb        # ミニマップ抽出
│   ├── 06_ward_batch_processing.ipynb  # ward検出
│   ├── 07_vision_score.ipynb      # 視界スコア学習
│   └── 08_win_prediction.ipynb    # 勝敗予測評価
├── scripts/                       # CLIスクリプト
├── models/                        # 学習済みモデル
│   └── best.pt                    # YOLOv8 ward検出
├── docs/                          # 技術ドキュメント
└── results/                       # 実験結果
```

## セットアップ

### 必要環境

- Python 3.7+
- Windows OS（LoLクライアント動作のため）
- NVIDIA GPU（推奨、YOLO推論高速化）
- League of Legendsクライアント

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/pyLoL-v2.git
cd pyLoL-v2

# 仮想環境の作成・有効化
python -m venv .venv
.venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt

# パッケージのインストール（開発モード）
python setup.py develop

# GPU使用時（オプション）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### APIキーの設定

1. [Riot Developer Portal](https://developer.riotgames.com)でAPIキーを取得
2. `RiotAPI.env`ファイルを作成し、キーを設定

## 使用方法

### 1. データ収集パイプライン

```bash
# Jupyter Notebookで順次実行
01_get_matchids.ipynb      # matchId収集
02_download_replays.ipynb  # リプレイダウンロード
04_run_client.ipynb        # ミニマップ抽出
```

### 2. ward検出・座標抽出

```bash
# バッチ処理（全試合）
python autoLeague/dataset/ward_tracker.py --all --hungarian
```

### 3. 視界スコア学習・勝敗予測

```bash
# Notebookで実行
07_vision_score.ipynb      # 視界スコアモデル学習
08_win_prediction.ipynb    # 勝敗予測・評価
```

## 技術的な工夫

### 1. ハンガリアン法によるwardマッチング

YOLO検出結果とRiot APIタイムラインデータのマッチングに最適割当問題を適用し、マッチング精度を85%まで向上。

### 2. グリッドベース特徴量

ミニマップを32×32グリッドに分割し、ward配置を空間的に表現。これにより位置情報を機械学習で扱いやすい形式に変換。

### 3. ヒートマップ学習による戦術スコア

ロジスティック回帰でグリッドごとの「勝利への寄与度」を学習。この重みを用いてward配置の戦術的価値を定量化。

## データセット

- **サーバー**: JP（日本）
- **ティア**: Diamond帯
- **試合数**: 92試合
- **パッチ**: 25.S1.2
- **評価手法**: Leave-One-Out Cross Validation

## 制約事項

- `.rofl`ファイルは現行パッチでのみ再生可能
- Riot APIの開発キーはレート制限あり（20 req/sec）
- Windows環境でのみ動作（LoLクライアント依存）

## 今後の課題

- [ ] データセット拡張（数百試合規模での検証）
- [ ] 時間帯別ヒートマップの最適化
- [ ] チャンピオン座標との統合
- [ ] LightGBM/XGBoostなど他モデルの検討

## ドキュメント

- [Phase 3: ward座標抽出仕様書](docs/phase3_ward_coordinate_extraction.md)
- [Phase 5: 視界スコア設計](docs/phase5_vision_score.md)
- [Phase 6: 分析レポート](results/phase6_analysis_report.md)


## 参考文献
- WARDS: Modelling the Worth of Vision in MOBA’s
- https://github.com/league-of-legends-replay-extractor/pyLoL
- Riot Games API Documentation
- YOLOv8 (Ultralytics)
