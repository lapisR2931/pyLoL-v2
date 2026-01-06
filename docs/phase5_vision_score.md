# Phase 5: 視界スコア指標設計書

## 1. 概要

### 1.1 目的
ward座標データを勝率予測モデルへの入力特徴量として活用し、予測精度の向上を目指す。

### 1.2 アプローチ
- **データドリブン**: 事前のエリア分割ではなく、生座標をグリッド化してモデルに学習させる
- **解釈可能性**: 学習後にヒートマップで「どの座標×時間帯が重要か」を可視化

### 1.3 入出力
```
入力: 時間帯別wardグリッド (時間帯数, H, W, チーム数)
出力: 勝敗予測 (0 or 1)
副産物: 重要座標のヒートマップ（分析用）
```

---

## 2. アーキテクチャ

### 2.1 データフロー

```
wards_matched.csv (Phase 3出力)
        ↓
[Task A] グリッド特徴量生成
        ↓
ward_grids.npz (時間帯別グリッドデータ)
        ↓
[Task B] 勝敗ラベル付与
        ↓
dataset.npz (特徴量 + ラベル)
        ↓
[Task C] 予測モデル学習
        ↓
model.pt (学習済みモデル)
        ↓
[Task D] ヒートマップ可視化
        ↓
heatmaps/ (分析結果画像)
```

### 2.2 ディレクトリ構造

```
autoLeague/
└── scoring/
    ├── __init__.py
    ├── grid_generator.py    # Task A: グリッド生成
    ├── dataset_builder.py   # Task B: データセット構築
    ├── predictor.py         # Task C: 予測モデル
    └── visualizer.py        # Task D: 可視化

scripts/
├── build_vision_dataset.py  # Task A+B 実行スクリプト
├── train_vision_model.py    # Task C 実行スクリプト
└── visualize_vision_heatmap.py  # Task D 実行スクリプト
```

---

## 3. 実装タスク（並列実行可能）

### Task A: グリッド特徴量生成
**ファイル**: `autoLeague/scoring/grid_generator.py`

**依存**: なし（並列実行可能）

**入力**:
- `wards_matched.csv` (各試合フォルダ内)

**出力**:
- `ward_grid.npz` (各試合フォルダ内)

**仕様**:
```python
# グリッド設定
GRID_SIZE = 32  # 512px / 32 = 16px単位
TIME_PHASES = [
    (0, 10 * 60 * 1000),      # 0-10分
    (10 * 60 * 1000, 20 * 60 * 1000),  # 10-20分
    (20 * 60 * 1000, None),   # 20分以降
]

# 出力形状
# blue_grid: (3, 32, 32) - 各セルはwardが存在した秒数
# red_grid: (3, 32, 32)
```

**主要関数**:
```python
def generate_ward_grid(wards_csv_path: Path) -> dict:
    """
    wards_matched.csvからグリッド特徴量を生成

    Returns:
        {
            "blue": np.ndarray (3, 32, 32),
            "red": np.ndarray (3, 32, 32),
            "match_id": str
        }
    """
```

---

### Task B: データセット構築
**ファイル**: `autoLeague/scoring/dataset_builder.py`

**依存**: Task A

**入力**:
- `ward_grid.npz` (各試合)
- `data/timeline/JP1_*.json` (勝敗情報)

**出力**:
- `data/vision_dataset.npz`

**仕様**:
```python
# データセット形状
X: (N, 2, 3, 32, 32)  # N試合, 2チーム, 3時間帯, 32x32グリッド
y: (N,)  # 勝敗ラベル (1=Blue勝利, 0=Red勝利)
match_ids: (N,)  # 試合ID
```

**主要関数**:
```python
def build_dataset(
    dataset_dir: Path,
    timeline_dir: Path,
    output_path: Path
) -> None:
    """
    全試合のグリッドデータと勝敗ラベルを統合
    """

def get_winner_from_timeline(timeline_path: Path) -> str:
    """
    タイムラインJSONから勝者チームを取得

    Returns:
        "blue" or "red"
    """
```

---

### Task C: 予測モデル学習
**ファイル**: `autoLeague/scoring/predictor.py`

**依存**: Task B

**入力**:
- `data/vision_dataset.npz`

**出力**:
- `models/vision_predictor.pt`
- `models/vision_predictor_metrics.json`

**モデル選択肢**:

1. **ベースライン: ロジスティック回帰**
   - グリッドをflattenして入力
   - 解釈しやすい（重みがそのまま重要度）

2. **発展版: 浅いCNN**
   - 空間的パターンを学習可能
   - Grad-CAMで重要領域を可視化

**主要クラス**:
```python
class VisionPredictor:
    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: "logistic" or "cnn"
        """

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """学習してmetricsを返す"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""

    def get_feature_importance(self) -> np.ndarray:
        """特徴量重要度を取得 (グリッド形状で返す)"""
```

---

### Task D: ヒートマップ可視化
**ファイル**: `autoLeague/scoring/visualizer.py`

**依存**: Task C

**入力**:
- `models/vision_predictor.pt`
- ミニマップ背景画像（オプション）

**出力**:
- `heatmaps/importance_phase{0,1,2}.png`
- `heatmaps/importance_combined.png`

**主要関数**:
```python
def visualize_importance_heatmap(
    importance: np.ndarray,  # (3, 32, 32)
    output_dir: Path,
    minimap_bg: Optional[Path] = None
) -> None:
    """
    特徴量重要度をヒートマップとして可視化

    - 時間帯別に3枚
    - 全時間帯を統合した1枚
    """
```

---

## 4. 並列実装ガイド

### 4.1 依存関係

```
Task A (グリッド生成)  ──→  Task B (データセット構築)  ──→  Task C (モデル学習)  ──→  Task D (可視化)
```

### 4.2 並列実行可能な組み合わせ

| チャット1 | チャット2 | 備考 |
|-----------|-----------|------|
| Task A | - | 最初に実行 |
| Task B | Task D (インターフェース定義のみ) | Task Aの出力形式確定後 |
| Task C | Task D (実装) | Task Bの出力形式確定後 |

### 4.3 インターフェース定義（先に合意が必要）

**グリッドデータ形式** (`ward_grid.npz`):
```python
{
    "blue": np.ndarray,  # shape: (3, 32, 32), dtype: float32
    "red": np.ndarray,   # shape: (3, 32, 32), dtype: float32
    "match_id": str
}
```

**データセット形式** (`vision_dataset.npz`):
```python
{
    "X": np.ndarray,      # shape: (N, 2, 3, 32, 32), dtype: float32
    "y": np.ndarray,      # shape: (N,), dtype: int32
    "match_ids": list     # length: N
}
```

**重要度形式**:
```python
importance: np.ndarray  # shape: (3, 32, 32), dtype: float32
# 正の値: Blue有利に寄与
# 負の値: Red有利に寄与
```

---

## 5. 実装優先順位

1. **Phase 5A**: Task A (グリッド生成) - 必須
2. **Phase 5B**: Task B (データセット構築) - 必須
3. **Phase 5C**: Task C (予測モデル) - 必須
4. **Phase 5D**: Task D (可視化) - オプション（分析用）

---

## 6. 追加検討事項

### 6.1 データ量の懸念
- 現在の試合数が少ない場合、モデルの汎化性能に影響
- 対策: シンプルなモデル（ロジスティック回帰）から開始

### 6.2 時間帯の粒度
- 初期設定: 0-10分 / 10-20分 / 20分以降
- 効果を見て調整（5分刻みなど）

### 6.3 ward種別の分離
- 初期: stealth_ward + control_ward を統合
- 発展: 別チャンネルとして分離

---

## 7. 参照ドキュメント

- [Phase 3: ward座標抽出仕様書](phase3_ward_coordinate_extraction.md)
- [wardType分類](ward_type_classification.md)
