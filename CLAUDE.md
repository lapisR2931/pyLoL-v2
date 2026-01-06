# CLAUDE.md - pyLoL v2 プロジェクトガイド
- このファイルはClaude Codeがプロジェクトを理解するためのドキュメントです


## プロジェクト概要

このプロジェクトはLoLリプレイからデータを抽出し、視界スコア指標を作成することで勝率予測精度の向上を目指します。

### 最終目標
**ward視界スコア指標の作成** → LoLリアルタイム勝率予測の精度向上 + 戦術評価

wards論文からインスパイアを受け、以下の要素を考慮した指標を作成する：
- 視界の範囲の有効性（どこが見えているか）
- 時間帯との関連性（いつ視界があるか）
- 戦術的価値（その視界が何を防ぐ/可能にするか）

### 中間目標（フェーズ別）
| Phase | 目標 | 状態 |
|-------|------|------|
| 1 | roflファイル収集パイプラインの作成 | 完了 |
| 2 | ward検出モデルの構築 | 完了 |
| 3 | ward座標抽出パイプライン構築 | 完了 |
| 4 | チャンピオントラッキングモデルの試行 | 未着手 |
| 5 | 視界スコア指標の設計・実装 | 未着手 |
| 6 | 勝率予測モデルへの統合・評価 | 未着手 |

## User prompt
Talk in Japanese.
Use polite Japanese (です・ます調) sentence endings in all responses.

You are a helpful, concise, and knowledgeable assistant. Follow these guidelines when responding:
- Use clear, precise, and professional language.
- Do not use emojis or emotive punctuation.
- Do not compliment, praise, or encourage the user.
- Prioritize factual accuracy and logical consistency.
- Apply rigorous critical thinking to user requests and assertions.
- Use bullet points or numbered lists where appropriate.
- Avoid unnecessary repetition or verbosity.
- If a request is unclear, ask for clarification before responding.
- Strengthen prompting to encourage claude to enter plan mode more often


## ディレクトリ構造

```
pyLoL-v2/
├── autoLeague/                    # メインパッケージ
│   ├── dataset/                   # データ取得・生成モジュール
│   │   ├── generator.py           # DataGenerator - matchId取得
│   │   ├── downloader.py          # ReplayDownloader - .roflダウンロード
│   │   ├── riotapi.py             # RiotAPI - 試合データ取得
│   │   ├── ward_tracker.py        # WardTracker - タイムライン統合
│   │   ├── calculator.py          # データ計算
│   │   └── preprocessor.py        # 前処理
│   ├── replays/                   # リプレイ操作モジュール
│   │   ├── scraper.py             # ReplayScraper - クライアント操作・ミニマップ抽出
│   │   └── editor.py              # リプレイ編集
│   ├── detection/                 # 検出モジュール（Phase 3〜4用）
│   │   └── ward_detector.py       # ward検出クラス（未実装）
│   ├── scoring/                   # 視界スコアモジュール（Phase 5用）
│   │   └── vision_score.py        # 視界スコア計算（未実装）
│   ├── preprocess/                # 画像/OCR前処理
│   │   └── ocr_center_window.py   # KDA/CS抽出用OCR
│   ├── utils/                     # 共通ユーティリティ
│   │   ├── csv_utils.py           # CSV操作
│   │   └── patch_info.py          # パッチ情報取得
│   └── bin/                       # バイナリ/ユーティリティ
│       └── utils.py
├── notebooks/                     # Jupyter Notebooks
│   ├── 01_get_matchids.ipynb      # Step1: matchId取得
│   ├── 02_download_replays.ipynb  # Step2: .roflダウンロード
│   ├── 03_filtering_replays.ipynb # Step3: フィルタリング
│   ├── 04_run_client.ipynb        # Step4: リプレイ再生・データ抽出
│   └── 05_ward_inference.ipynb    # Step5: ward検出推論
├── scripts/                       # スタンドアロンスクリプト
│   ├── batch_inference_wards.py   # バッチ推論＋クラスタリング
│   ├── compare_matching_methods.py # 貪欲法vsハンガリアン法比較
│   ├── visualize_matching_comparison.py # マッチング結果可視化
│   ├── analyze_river_sight.py     # リバーサイト座標分析
│   ├── train_yolo_ward.py         # YOLOv8学習スクリプト
│   ├── prepare_yolo_dataset.py    # JSON→YOLO形式変換
│   ├── export_onnx.py             # ONNXエクスポート
│   ├── visualize_wards.py         # ward可視化
│   ├── visualize_ward_timeline.py # タイムラインward可視化
│   ├── visualize_ward_by_player.py # プレイヤー別ward分析
│   ├── analyze_undefined_wards.py # UNDEFINED wardType分析
│   └── get_match_participants.py  # 試合参加者情報取得
├── models/                        # 学習済みモデル
│   └── best.pt                    # YOLOv8 ward検出モデル v4
├── matchids/                      # 取得したmatchIdのCSV保存先
├── data/                          # 処理済みデータ保存先（gitignore対象）
├── docs/                          # ドキュメント
│   ├── ward_detection_planB.md    # Phase 2: ward検出Plan B仕様書
│   ├── phase3_ward_coordinate_extraction.md  # Phase 3: ward座標抽出仕様書
│   ├── ward_type_classification.md # Riot API wardType分類
│   └── project_status_20260104.md # プロジェクト進捗・引き継ぎ
├── setup.py
├── requirements.txt
├── RiotAPI.env                    # APIキー（gitignore対象）
├── .gitignore
└── CLAUDE.md
```

### 外部ディレクトリ（gitignore対象）
| パス | 用途 |
|------|------|
| `C:\dataset_20260101` | ミニマップキャプチャ画像（26試合、旧データ） |
| `C:\dataset_20260105` | ミニマップキャプチャ画像（frame_timestamps.csv対応、現行） |
| `C:\dataset_annotation` | アノテーション作業用（X-AnyLabeling） |
| `C:\dataset_yolo` | YOLO学習用データセット・モデル出力 |


## 開発環境

- **Python**: 3.7+ （.venv使用）
- **OS**: Windows（LoLクライアントがWindows専用のため）
- **必須ソフトウェア**: League of Legendsクライアント

```bash
# セットアップ
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup.py develop
```


## よく使うコマンド

```bash
# パッケージを開発モードでインストール
python setup.py develop

# 仮想環境の有効化
.venv\Scripts\activate
```


## ワークフロー

データ収集から抽出までのパイプライン：

1. **01_get_matchids.ipynb** - Riot APIでmatchId収集
2. **02_download_replays.ipynb** - .roflファイルダウンロード
3. **03_filtering_replays.ipynb** - データフィルタリング
4. **04_run_client.ipynb** - リプレイ再生・ミニマップ抽出
5. **05_ward_inference.ipynb** - ward検出推論テスト


## API/外部サービス

### Riot API
- **取得先**: https://developer.riotgames.com
- **環境変数**: RiotAPI.env参照
- **レート制限**:
  - 開発キー: 20 requests/sec, 100 requests/2min
  - 本番キー: より高い制限
- **使用エンドポイント**:
  - `/lol/league/v4/entries` - ティア別プレイヤー取得
  - `/lol/match/v5/matches/{matchId}` - 試合情報
  - `/lol/match/v5/matches/{matchId}/timeline` - タイムラインデータ

### LCU API（LoL Client Update API）
- LoLクライアントのローカルAPI
- リプレイダウンロード・再生制御に使用
- **主要エンドポイント**:
  - `POST /lol-replays/v1/rofls/{gameId}/download/graceful` - リプレイダウンロード
  - `POST /lol-replays/v1/rofls/{gameId}/watch` - リプレイ起動

### Replay API（ゲーム内API）
- リプレイ再生中のみアクセス可能（port 2999）
- 認証不要
- **主要エンドポイント**:
  - `GET /replay/playback` - 再生状態・ゲーム長取得
  - `POST /replay/playback` - 再生制御
  - `POST /replay/render` - 描画設定


## 既知の問題・注意点

- **パッチ依存**: .roflファイルは現行パッチでのみ再生可能
- **API制限**: Riot APIのレート制限に注意
- **クライアント必須**: notebook 02, 04はLoLクライアント起動状態が必要


## 進捗管理

### 完了したPhase

**Phase 1: データ収集パイプライン** - 完了
- notebook 01〜04の実装完了
- LCU API経由リプレイ起動実装

**Phase 2: ward検出モデル** - 完了
- Plan B（実データアノテーション）方式で実装
- YOLOv8n v4モデル: `models/best.pt`
- 学習履歴: `C:\dataset_yolo\runs\ward_detection_v4\`

**Phase 3: ward座標抽出パイプライン** - 完了（2026-01-05更新）
- タイムラインデータとYOLO検出結果のマッチング実装
- `autoLeague/dataset/ward_tracker.py` - WardTrackerクラス
- `scripts/batch_inference_wards.py` - `--with-timeline`、`--hungarian`オプション追加
- `autoLeague/replays/scraper.py` - frame_timestamps.csv記録機能追加
- 出力: `wards_matched.csv`
- マッチング率: 約85%（ハンガリアン法使用時）
- 主要設定:
  - `FRAME_TOLERANCE=10`（約5秒）
  - リバーサイトフィルタリング: 座標(362,332)/(151,178) + 持続時間75~93秒
- コマンド例:
  ```bash
  # 全試合処理（ハンガリアン法）
  python autoLeague/dataset/ward_tracker.py --all --hungarian
  # 1試合処理
  python autoLeague/dataset/ward_tracker.py --match JP1-555621265 --hungarian
  ```
- 仕様書: `docs/phase3_ward_coordinate_extraction.md`

### 今後の予定

**Phase 4: ※スキップ　チャンピオントラッキング**
必要性が低いためスキップ
- [ ] ミニマップからチャンピオンアイコン検出
- [ ] 移動経路抽出・座標記録

**Phase 5: 視界スコア指標** - 設計完了（2026-01-06）
- 仕様書: `docs/phase5_vision_score.md`
- アプローチ: グリッドベース特徴量 + 勝敗予測モデル
- 実装タスク（並列実行可能）:
  - [ ] Task A: グリッド特徴量生成 (`autoLeague/scoring/grid_generator.py`)
  - [ ] Task B: データセット構築 (`autoLeague/scoring/dataset_builder.py`)
  - [ ] Task C: 予測モデル学習 (`autoLeague/scoring/predictor.py`)
  - [ ] Task D: ヒートマップ可視化 (`autoLeague/scoring/visualizer.py`)
- 依存関係: A → B → C → D

**Phase 6: 勝率予測モデル統合**
- [ ] 既存勝率予測モデルの調査
- [ ] 視界スコア特徴量の追加


## 開発ルール
- NotebookのコードセルにはCell番号コメントを記載
- APIキーはRiotAPI.envファイルで管理
- Windowsパスは Path(r"C:\...") 形式で記述
- matchidsフォルダのCSVはパッチバージョンをファイル名に含める
- 大規模データ（data/フォルダ）は.gitignoreで除外
