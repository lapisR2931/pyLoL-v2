# Ward戦術スコア評価結果

## 概要

ヒートマップ重要度マップを活用した戦術スコア特徴量の評価結果。

**評価日**: 2026-01-07
**サンプル数**: 92試合（Blue勝利: 45, Red勝利: 47）
**評価手法**: Leave-One-Out 交差検証

## 評価結果

### 10分時点

| モデル | Accuracy | AUC | LogLoss | 特徴量数 |
|--------|----------|-----|---------|----------|
| baseline | 59.8% | 0.648 | 0.666 | 20 |
| baseline_riot | 68.5% | 0.710 | 0.636 | 23 |
| baseline_grid | 59.8% | 0.648 | 0.675 | 26 |
| **baseline_tactical** | **77.2%** | **0.826** | **0.519** | 26 |

### 20分時点

| モデル | Accuracy | AUC | LogLoss | 特徴量数 |
|--------|----------|-----|---------|----------|
| baseline | 80.4% | 0.901 | 0.401 | 20 |
| baseline_riot | 84.8% | 0.907 | 0.386 | 23 |
| baseline_grid | 80.4% | 0.884 | 0.420 | 26 |
| **baseline_tactical** | **88.0%** | **0.945** | **0.320** | 26 |

## 改善度

### vs baseline（標準特徴量のみ）

| 時点 | Accuracy改善 | AUC改善 |
|------|-------------|---------|
| 10分 | +17.4pt | +0.178 |
| 20分 | +7.6pt | +0.044 |

### vs baseline_riot（Riot visionScore追加）

| 時点 | Accuracy改善 | AUC改善 |
|------|-------------|---------|
| 10分 | +8.7pt | +0.116 |
| 20分 | +3.2pt | +0.038 |

### vs baseline_grid（グリッド統計量追加）

| 時点 | Accuracy改善 | AUC改善 |
|------|-------------|---------|
| 10分 | +17.4pt | +0.178 |
| 20分 | +7.6pt | +0.061 |

## 戦術スコア特徴量

| 特徴量名 | 説明 |
|----------|------|
| blue_placement_score | Blueチームの設置スコア累積 |
| red_placement_score | Redチームの設置スコア累積 |
| blue_deny_score | Blueチームの破壊スコア累積 |
| red_deny_score | Redチームの破壊スコア累積 |
| placement_score_diff | 設置スコア差（Blue - Red） |
| deny_score_diff | 破壊スコア差（Blue - Red） |

### スコアリングルール

- **設置スコア**: 自チームに有利な位置にwardを置くと獲得
  - Blue: 正の重要度セル → 高スコア
  - Red: 負の重要度セル → 高スコア
- **破壊スコア**: 敵wardを破壊した側のチームに付与
  - 敵チームに有利な位置のward破壊 → 高スコア

| 重要度の絶対値 | スコア |
|---------------|--------|
| >= 0.03 | 2点 |
| 0.01 ~ 0.03 | 1点 |
| < 0.01 | 0点 |

## 結論

1. **戦術スコア特徴量が最も高い予測精度を達成**
   - 特に10分時点での改善が顕著（+17.4pt）
   - 試合早期のward配置の質的評価に有効

2. **Riot visionScoreを上回る性能**
   - 単純なward設置数ではなく、位置の戦術的価値を考慮
   - ヒートマップベースの評価が有効に機能

3. **グリッド統計量よりも効果的**
   - 生のグリッド情報より、重要度で重み付けしたスコアが有効

## 実装ファイル

- `autoLeague/scoring/tactical_scorer.py` - 戦術スコア計算
- `autoLeague/prediction/config.py` - 特徴量定義
- `autoLeague/prediction/feature_extractor.py` - 特徴量抽出
- `autoLeague/prediction/baseline_predictor.py` - 予測モデル

## 生成ファイル

- `C:\dataset_20260105\JP1-*/ward_tactical_scores.csv` - 各試合の戦術スコア
- `data/prediction_dataset.npz` - 学習用データセット
- `results/tactical_score_evaluation.json` - 評価結果（JSON）
