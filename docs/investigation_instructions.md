# Ward座標抽出パイプライン - 追加調査指示書

**作成日**: 2026-01-05
**目的**: timeline_only問題の根本原因特定と改善策の実装

---

## 背景

pyLoL-v2プロジェクトのPhase 3（ward座標抽出パイプライン）において、タイムラインイベントとYOLO検出結果のマッチング処理を行っている。現在のマッチング率は85.6%で、18件がtimeline_only（マッチング失敗）となっている。

調査の結果、主要な原因は以下と判明：
- **チーム誤分類（9件/50%）**: YOLOがwardのチーム（blue/red）を誤認識
- **tolerance境界（3件/17%）**: わずかにフレーム許容範囲外
- **その他（6件/33%）**: 検出漏れ等

---

## 調査タスク

### タスク1: チーム誤分類の目視確認

**目的**: YOLOがチームを誤分類したwardの実際の画像を確認し、モデル改善の方向性を決定

**対象フレーム**:
```
C:\dataset_20260101\JP1-555841779\0\
├── 2100.png  # ward_id=74: BLUE_TRINKET blue → redと誤検出
├── 2111.png  # ward_id=77: CONTROL_WARD red → blueと誤検出
├── 2151.png  # ward_id=78: CONTROL_WARD red → blueと誤検出
├── 2336.png  # ward_id=87: CONTROL_WARD red → blueと誤検出
├── 2445.png  # ward_id=90: BLUE_TRINKET blue → redと誤検出
├── 2774.png  # ward_id=103: SIGHT_WARD red → blueと誤検出
├── 2817.png  # ward_id=104: SIGHT_WARD red → blueと誤検出
├── 2927.png  # ward_id=108: SIGHT_WARD blue → redと誤検出
└── 3215.png  # ward_id=122: SIGHT_WARD blue → redと誤検出
```

**確認ポイント**:
1. wardの色は青/緑（blue）か、赤/ピンク（red）か
2. wardの位置（マップ上のどこに配置されているか）
3. 背景や周囲の要素がチーム判定に影響していないか

**出力**: 各フレームの所見をまとめたレポート

---

### タスク2: 検出漏れの確認

**目的**: YOLOが検出できなかったwardが実際に画面に表示されているか確認

**対象フレーム**:
```
C:\dataset_20260101\JP1-555841779\0\
├── 2314.png  # ward_id=86: CONTROL_WARD blue (差-228フレーム)
├── 2379.png  # ward_id=89: SIGHT_WARD red (差-121フレーム)
└── 1839.png, 1853.png, 2109.png, 3236.png  # その他
```

**確認ポイント**:
1. 該当フレームにwardが表示されているか
2. 表示されている場合、なぜ検出されなかったか（小さい、隠れている等）
3. タイムスタンプのずれにより、別のフレームを確認すべきか

---

### タスク3: tolerance拡大の影響評価

**目的**: toleranceを30→50に拡大した場合の影響を評価

**実行コマンド**:
```bash
cd c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL-v2
.venv\Scripts\python -m autoLeague.dataset.ward_tracker --match JP1-555841779 --tolerance 50
```

**確認ポイント**:
1. マッチング率の変化
2. 誤マッチング（間違ったwardと紐付け）が発生していないか
3. timeline_only/detection_onlyの件数変化

---

### タスク4: チーム判定緩和オプションの実装検討

**目的**: チームが不一致でもマッチング候補として扱うオプションの実装

**実装案**:
```python
# ward_tracker.py の find_matching_candidates 関数を修正

def find_matching_candidates(
    event: TimelineWardEvent,
    detected_wards: List[DetectedWard],
    frame_tolerance: int,
    ignore_team: bool = False  # 新オプション
) -> List[DetectedWard]:
    for ward in detected_wards:
        # チームチェック（オプションで無効化可能）
        if not ignore_team and ward.team != event.team:
            continue
        # ... 残りのロジック
```

**注意点**:
- ignore_team=True の場合、マッチング後にタイムラインのチーム情報を優先して上書きする必要がある
- 誤マッチングのリスクが増加するため、ward_typeの一致は必須とする

---

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `autoLeague/dataset/ward_tracker.py` | マッチングロジック本体 |
| `C:\dataset_20260101\JP1-555841779\wards_matched.csv` | マッチング結果 |
| `C:\dataset_20260101\JP1-555841779\wards.csv` | YOLO検出結果（クラスタリング済み） |
| `data/timeline/JP1_555841779.json` | タイムラインデータ |
| `docs/investigation_timeline_only.md` | 調査レポート |

---

## 期待される成果

1. チーム誤分類の根本原因の特定
2. tolerance最適値の決定
3. マッチング率90%以上への改善策の提案

---

## 備考

- YOLOモデル: `C:\dataset_yolo\runs\ward_detection_v3\weights\best.pt`
- モデルのクラス: stealth_ward, stealth_ward_enemy, control_ward, control_ward_enemy
- 現在のマッチング許容範囲: -30 ~ +90 フレーム（frame_tolerance=30）
