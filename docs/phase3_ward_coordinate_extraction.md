# Phase 3: Ward座標抽出パイプライン仕様書

## 概要

Riot APIのtimelineデータ（WARD_PLACED/WARD_KILLイベント）とYOLO検出結果をマッチングし、各wardに一意のIDと座標を付与するパイプライン。

## 実装状態: 完了

- 実装日: 2026-01-04
- 最終更新: 2026-01-05（frame_timestamps.csv対応、FRAME_TOLERANCE調整）
- テスト結果: JP1-555621265で検証済み（マッチング率83.2%）

---

## アーキテクチャ

```
[タイムラインJSON] ──┐
    (data/timeline)  │
                     ├──> [WardTracker] ──> wards_matched.csv
[wards.csv] ─────────┘
    (検出結果)
```

## ファイル構成

### 新規作成ファイル

| ファイル | 説明 |
|---------|------|
| `autoLeague/dataset/ward_tracker.py` | wardマッチングロジック |

### 更新ファイル

| ファイル | 変更内容 |
|---------|---------|
| `scripts/batch_inference_wards.py` | `--with-timeline`オプション追加 |

---

## フレーム↔ゲーム時間マッピング

### 問題点（2026-01-05発見）

当初の固定値計算（376ms/frame）では、実際のキャプチャとタイムラインの間に大きなずれが発生することが判明。

**原因**: `time.sleep(0.047)` + キャプチャ処理のオーバーヘッドにより、実際の1ループは約0.063秒かかる。

| 項目 | 想定値 | 実測値（JP1-555841779） |
|------|--------|------------------------|
| MS_PER_FRAME | 376ms | 503ms |
| 1ループの実時間 | 0.047秒 | 0.063秒 |

### 現在の計算式（動的計算）

試合ごとにフレーム数とゲーム時間から係数を算出する方式に変更。

```python
# 試合ごとに動的に計算
total_frames = len(list(frame_dir.glob("*.png")))
game_duration_ms = timeline_data["info"]["frames"][-1]["timestamp"]
MS_PER_FRAME = game_duration_ms / total_frames

# タイムスタンプ（ms）→ フレーム番号
frame = timestamp_ms / MS_PER_FRAME

# フレーム番号 → タイムスタンプ（ms）
timestamp_ms = frame * MS_PER_FRAME
```

### キャプチャ時にゲーム内時間を記録（実装済み: 2026-01-05）

**方式**: scraper.pyのキャプチャループでReplay APIからゲーム内時間を取得し、フレーム番号と一緒にCSVに記録。

```python
# scraper.py でのキャプチャ時（run_client_lcu メソッド）
resp = requests.get('https://127.0.0.1:2999/replay/playback', verify=False, timeout=0.1)
current_game_time_ms = int(resp.json().get('time', 0) * 1000)
# frame_number と game_time_ms を frame_timestamps.csv に保存
```

**出力ファイル**: `{match_dir}/frame_timestamps.csv`
```csv
frame_number,game_time_ms
0,5000
1,5503
2,6006
...
```

**メリット**:
- フレーム単位で正確なゲーム内時間を取得可能
- 環境依存のオーバーヘッド変動を完全に吸収
- 動的計算より高精度なマッチングが期待できる

**デメリット**:
- 既存データには適用不可（再キャプチャが必要）
- API呼び出しのオーバーヘッド（約1-5ms/フレーム、実用上問題なし）

**ward_tracker.pyでの使用**:
- `frame_timestamps.csv`が存在する場合は自動的に使用
- 存在しない場合は従来の動的MS_PER_FRAME計算にフォールバック

### 例（動的計算の場合）

JP1-555841779の場合（MS_PER_FRAME = 503ms）:

| ゲーム時間 | タイムスタンプ(ms) | フレーム番号 |
|-----------|-------------------|-------------|
| 1:00 | 60,000 | 119 |
| 5:00 | 300,000 | 597 |
| 10:00 | 600,000 | 1,193 |
| 20:00 | 1,200,000 | 2,386 |

---

## タイムラインデータ構造

### 保存場所

`data/timeline/JP1_{match_id}.json`

### WARD_PLACEDイベント

```json
{
  "creatorId": 5,
  "timestamp": 55785,
  "type": "WARD_PLACED",
  "wardType": "YELLOW_TRINKET"
}
```

### WARD_KILLイベント

```json
{
  "killerId": 10,
  "timestamp": 657595,
  "type": "WARD_KILL",
  "wardType": "CONTROL_WARD"
}
```

### wardType一覧

| wardType | 説明 |
|----------|------|
| YELLOW_TRINKET | トリンケットワード |
| SIGHT_WARD | サポートアイテムのワード |
| CONTROL_WARD | コントロールワード |
| UNDEFINED | 破壊イベント等で種別不明 |

### チーム判定

- `creatorId 1-5` → Blue team
- `creatorId 6-10` → Red team

---

## 検出クラス↔タイムラインマッピング

| 検出クラス | チーム | 対応wardType |
|-----------|--------|-------------|
| stealth_ward | Blue | YELLOW_TRINKET, SIGHT_WARD |
| stealth_ward_enemy | Red | YELLOW_TRINKET, SIGHT_WARD |
| control_ward | Blue | CONTROL_WARD |
| control_ward_enemy | Red | CONTROL_WARD |

---

## マッチングアルゴリズム

### 処理フロー

1. タイムラインのward設置イベントをタイムスタンプ順にソート
2. 各設置イベントに対して:
   - 設置タイムスタンプをフレーム番号に変換（frame_expected）
   - frame_expected ± tolerance内で新規出現したwardクラスタを検索
   - 同一チーム・wardTypeの中で最も信頼度が高いものを採用
3. マッチングしなかった検出結果は`detection_only`として記録
4. マッチングしなかったタイムラインイベントは`timeline_only`として記録

### 設定値

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| FRAME_TOLERANCE | 10 | マッチング許容フレーム数（約5秒） |
| IMAGE_SIZE | 512 | ミニマップサイズ（px） |

---

## 出力形式

### wards_matched.csv

```csv
ward_id,timeline_ward_id,class_name,ward_type,team,x_pixel,y_pixel,x_normalized,y_normalized,frame_start,frame_end,confidence_avg,creator_id,timestamp_placed,timestamp_killed,match_status
1,1,stealth_ward,YELLOW_TRINKET,blue,453,447,0.886692,0.874573,202,228,0.4656,2,49605,,matched
2,2,stealth_ward,YELLOW_TRINKET,blue,,,,,148,,,5,55785,,timeline_only
```

### カラム説明

| カラム | 型 | 説明 |
|--------|-----|------|
| ward_id | int | 統合後の一意ID |
| timeline_ward_id | int/null | タイムラインイベントID |
| class_name | str | YOLO検出クラス名 |
| ward_type | str | タイムラインのwardType |
| team | str | blue/red |
| x_pixel, y_pixel | int | ピクセル座標（512x512） |
| x_normalized, y_normalized | float | 正規化座標（0-1） |
| frame_start, frame_end | int | 検出フレーム範囲 |
| confidence_avg | float | 平均検出信頼度 |
| creator_id | int/null | 設置者のparticipantId |
| timestamp_placed | int/null | 設置タイムスタンプ(ms) |
| timestamp_killed | int/null | 破壊タイムスタンプ(ms) |
| match_status | str | matched/detection_only/timeline_only |

---

## 使用方法

### 基本的な使用

```bash
# 1試合のみ処理（タイムライン統合あり）
python scripts/batch_inference_wards.py --match JP1-555621265 --with-timeline

# 全試合処理（タイムライン統合あり）
python scripts/batch_inference_wards.py --all --with-timeline
```

### WardTrackerを直接使用（既にwards.csvがある場合）

```bash
python -m autoLeague.dataset.ward_tracker --match JP1-555621265
python -m autoLeague.dataset.ward_tracker --all
```

### Pythonから使用

```python
from autoLeague.dataset.ward_tracker import WardTracker
from pathlib import Path

tracker = WardTracker(
    timeline_dir=Path("data/timeline"),
    dataset_dir=Path(r"C:\dataset_20260105")
)

# 1試合処理
result = tracker.process_match("JP1-555621265")

# 全試合処理
results = tracker.process_all()
```

---

## テスト結果

### JP1-555621265（最新: 2026-01-05、frame_timestamps.csv使用）

| 項目 | 値 |
|------|-----|
| タイムラインward設置（全体） | 268件 |
| フィルタリング除外 | 119件 |
| ├ creatorId=0 | 44件 |
| └ UNDEFINED | 75件 |
| 有効イベント | 149件 |
| YOLO検出ward | 184件 |
| 出力ward数 | 209件 |
| マッチング成功 | 124件 (83.2%) |
| 検出のみ（detection_only） | 60件 |
| タイムラインのみ（timeline_only） | 25件 |

### マッチング率改善の経緯

| バージョン | マッチング率 | 変更内容 |
|-----------|-------------|---------|
| 初期実装 | 51.0% | 固定MS_PER_FRAME=376ms |
| 動的計算導入 | 52.0% | 試合ごとにMS_PER_FRAMEを動的計算（tolerance=10） |
| frame_timestamps.csv導入 | **83.2%** | キャプチャ時にゲーム内時間を記録 |

### FRAME_TOLERANCE設定の根拠

wardがチャンピオンアイコンに隠れる時間は約5〜10秒程度。この可視ウィンドウ内で確実にキャプチャするため、FRAME_TOLERANCE=10（約5秒）を採用。

過度に大きなtoleranceは誤マッチングのリスクを高めるため、最小限に設定。

### フィルタリング対象

1. **creatorId=0**: 無効なparticipantId（イベントデータの欠損）
2. **UNDEFINED**: wardTypeが記録されていない（マッチング不可能）

### マッチング対象のwardType

| wardType | 説明 | YOLOクラス |
|----------|------|-----------|
| YELLOW_TRINKET | トリンケットワード | stealth_ward / stealth_ward_enemy |
| SIGHT_WARD | サポートアイテムのワード | stealth_ward / stealth_ward_enemy |
| BLUE_TRINKET | ファーサイトオルタネーション | stealth_ward / stealth_ward_enemy |
| CONTROL_WARD | コントロールワード | control_ward / control_ward_enemy |

### detection_onlyの考察

- detection_only（122件）が多い理由：
  1. 観戦モードでは全チームのwardが見えるため、タイムラインよりも多くのwardを検出
  2. タイムラインのwardTypeがUNDEFINEDでフィルタされたwardは、検出されてもマッチング対象外

---

## 制約事項

1. **Riot API制限**: WARD_PLACED/WARD_KILLイベントには座標情報がない
2. **視界の制約**: 敵wardは味方の視界に入らないと検出できない
3. **フレーム精度**: キャプチャ間隔（0.047秒 × speed）による誤差

---

## 今後の改善案

1. **マッチング精度向上**: tolerance値の調整、座標ベースの追加マッチング
2. **視界外ward推定**: チャンピオン位置と設置時刻から座標を推測
3. **KILL時刻の紐付け精度向上**: ward IDベースのトラッキング
