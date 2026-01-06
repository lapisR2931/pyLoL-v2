# プロジェクト進捗・引き継ぎドキュメント

**作成日**: 2026-01-04
**プロジェクト**: pyLoL-v2 - LoLリプレイからの視界スコア指標作成

---

## プロジェクト全体の進捗

| Phase | 内容 | 状態 |
|-------|------|------|
| 1 | roflファイル収集パイプライン | 完了 |
| 2 | ward検出モデル構築 | 完了 |
| **3** | **ward座標抽出パイプライン** | **完了** |
| 4 | チャンピオントラッキングモデル | 未着手 |
| 5 | 視界スコア指標の設計・実装 | 未着手 |
| 6 | 勝率予測モデルへの統合・評価 | 未着手 |

---

## Phase 3 完了サマリー

### 実装内容

1. **WardTrackerクラス**: タイムラインとYOLO検出のマッチング
2. **batch_inference_wards.py更新**: `--with-timeline`オプション追加
3. **出力形式**: `wards_matched.csv`

### 主要な技術的決定事項

| 項目 | 決定内容 |
|------|---------|
| フレーム変換式 | `frame = timestamp_ms / 376` |
| キャプチャ間隔 | 0.047秒（scraper.py） |
| リプレイ速度 | 8（固定） |
| マッチング許容フレーム | 30フレーム |

### テスト結果

- マッチング率: **57.3%**（110/192）
- 検出のみ: 190件、タイムラインのみ: 82件
- 出力総数: 382件

---

## 現在のデータ状況

### データセット

| 項目 | 値 |
|------|-----|
| 場所 | `C:\dataset_20260101` |
| 試合数 | 26試合 |
| progress.json | 26試合全て`success` |

### タイムラインデータ

| 項目 | 値 |
|------|-----|
| 場所 | `data/timeline/` |
| ファイル数 | 26ファイル |
| 形式 | `JP1_{match_id}.json` |

### ward検出モデル

| 項目 | 値 |
|------|-----|
| モデルパス | `C:\dataset_yolo\runs\ward_detection_v3\weights\best.pt` |
| アーキテクチャ | YOLOv8n |
| mAP50 | 0.983 |
| クラス | stealth_ward, stealth_ward_enemy, control_ward, control_ward_enemy |

---

## ディレクトリ構造（更新版）

```
pyLoL-v2/
├── autoLeague/
│   ├── dataset/
│   │   ├── generator.py       # matchId取得
│   │   ├── downloader.py      # .roflダウンロード
│   │   ├── riotapi.py         # Riot API操作
│   │   └── ward_tracker.py    # [NEW] wardマッチング
│   ├── replays/
│   │   └── scraper.py         # ミニマップ抽出
│   └── ...
├── scripts/
│   ├── batch_inference_wards.py  # [UPDATED] --with-timeline追加
│   ├── train_yolo_ward.py
│   └── ...
├── docs/
│   ├── ward_detection_planB.md           # Phase 2仕様書
│   ├── phase3_ward_coordinate_extraction.md  # [NEW] Phase 3仕様書
│   └── project_status_20260104.md        # [NEW] 本ドキュメント
├── data/
│   └── timeline/              # タイムラインJSON（26試合）
└── ...
```

---

## 外部ディレクトリ

| パス | 用途 |
|------|------|
| `C:\dataset_20260101` | ミニマップキャプチャ画像（26試合） |
| `C:\dataset_yolo` | YOLO学習用データセット・モデル |

---

## 次のアクション候補

### 即時実行可能

1. **全試合へのward座標抽出適用**
   ```bash
   python scripts/batch_inference_wards.py --all --with-timeline
   ```

2. **マッチング精度の分析**
   - `detection_only`wardの分布確認
   - `timeline_only`wardの原因調査

### Phase 4準備

1. **チャンピオントラッキングモデルの調査**
   - ミニマップ上のチャンピオンアイコン検出
   - 移動経路抽出・座標記録

### Phase 5準備

1. **wards論文の詳細調査**
   - 視界の有効性評価指標
   - 時間帯別・エリア別視界価値

---

## 重要な技術情報

### Riot API タイムラインイベント

```python
# WARD_PLACEDイベント
{
    "creatorId": 5,        # 設置者ID (1-5: Blue, 6-10: Red)
    "timestamp": 55785,    # ミリ秒
    "type": "WARD_PLACED",
    "wardType": "YELLOW_TRINKET"  # or SIGHT_WARD, CONTROL_WARD
}

# WARD_KILLイベント
{
    "killerId": 10,
    "timestamp": 657595,
    "type": "WARD_KILL",
    "wardType": "CONTROL_WARD"
}
```

### フレーム↔時間変換

```python
# 定数
CAPTURE_INTERVAL = 0.047  # 秒
REPLAY_SPEED = 8
MS_PER_FRAME = 376        # = 0.047 * 8 * 1000

# 変換
frame = timestamp_ms / 376
timestamp_ms = frame * 376
```

### 検出クラスとチーム

| 検出クラス | チーム | wardType |
|-----------|--------|----------|
| stealth_ward | Blue | YELLOW_TRINKET, SIGHT_WARD |
| stealth_ward_enemy | Red | YELLOW_TRINKET, SIGHT_WARD |
| control_ward | Blue | CONTROL_WARD |
| control_ward_enemy | Red | CONTROL_WARD |

---

## 既知の課題

1. **マッチング率57%**: 視界外ward、敵ward未発見が主因
2. **KILL時刻の紐付け**: 現在は近似マッチング（30秒以内）
3. **control_ward_enemy検出精度**: v3モデルで改善済みだが要監視

---

## 参照ドキュメント

- [Phase 2: ward検出Plan B仕様書](ward_detection_planB.md)
- [Phase 3: ward座標抽出パイプライン仕様書](phase3_ward_coordinate_extraction.md)
- [CLAUDE.md](../CLAUDE.md) - プロジェクト全体ガイド
