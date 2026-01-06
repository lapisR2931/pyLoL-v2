# Ward検出モデル実装仕様書（Plan B: 実データアノテーション方式）

作成日: 2026-01-02
ステータス: PoC完了（精度検証済み）

---

## 0. 進捗サマリー

### 実装完了項目（2026-01-03）

| 項目 | ステータス | 備考 |
|------|-----------|------|
| X-AnyLabelingセットアップ | 完了 | CPU版使用 |
| 100枚アノテーション | 完了 | 手動アノテーション |
| 追加50枚アノテーション | 完了 | 20分以降のフレーム、control_ward_enemy強化 |
| JSON→YOLO形式変換 | 完了 | scripts/prepare_yolo_dataset.py |
| YOLOv8学習（v3） | 完了 | 150枚、120 train / 30 val |
| ONNXエクスポート | 完了 | X-AnyLabeling連携用（v2） |
| 実データ推論テスト | 完了 | notebook [7] |

### モデル学習履歴

| バージョン | 学習データ | mAP50 | mAP50-95 | 備考 |
|-----------|-----------|-------|----------|------|
| v1 | 27枚 | - | - | 初期テスト、Auto Label用 |
| v2 | 100枚 | 0.942 | 0.741 | PoC本番モデル |
| v3 | 150枚 | **0.983** | **0.797** | 最終モデル（全クラス0.98以上） |

### v3モデル詳細精度

| クラス | Precision | Recall | mAP50 | v2比較 |
|--------|-----------|--------|-------|--------|
| stealth_ward | 0.967 | 0.968 | 0.983 | +0.015 |
| stealth_ward_enemy | 0.985 | 0.925 | 0.983 | +0.015 |
| control_ward | 0.982 | 0.899 | 0.985 | +0.033 |
| control_ward_enemy | 0.981 | 0.947 | 0.981 | **+0.099** |
| **全体** | **0.979** | **0.935** | **0.983** | +0.041 |

### v2→v3 改善サマリー

| 指標 | v2 (100枚) | v3 (150枚) | 改善 |
|------|-----------|-----------|------|
| mAP50 | 0.942 | 0.983 | +0.041 |
| mAP50-95 | 0.741 | 0.797 | +0.056 |
| Precision | 0.932 | 0.979 | +0.047 |
| Recall | 0.912 | 0.935 | +0.023 |

### 生成ファイル

| ファイル | パス |
|---------|------|
| 学習スクリプト | scripts/train_yolo_ward.py |
| データ変換スクリプト | scripts/prepare_yolo_dataset.py |
| ラベル修正スクリプト | scripts/fix_label_names.py |
| 追加画像選択スクリプト | scripts/select_additional_images.py |
| ONNXエクスポート | scripts/export_onnx.py |
| PyTorchモデル（v3） | C:\dataset_yolo\runs\ward_detection_v3\weights\best.pt |
| PyTorchモデル（v2） | C:\dataset_yolo\runs\ward_detection_v2\weights\best.pt |
| ONNXモデル（v2） | C:\dataset_yolo\runs\ward_detection_v2\weights\best.onnx |
| X-AnyLabeling設定 | C:\dataset_annotation\ward_detection_model.yaml |

### 次のアクション

1. **ward座標抽出パイプライン構築**: 実運用に向けた統合
2. **バッチ推論**: C:\dataset全体（89試合）への適用
3. **ward配置ヒートマップ生成**: 統計分析

---

## 1. プロジェクト概要

### 1.1 目的
- YOLOv8を使用してミニマップ（512x512px）から4クラスのward座標を検出
- Roboflowを使用した実データアノテーションでDomain Gapを解消
- 100枚以下の少量データでPoC（概念実証）を実施

### 1.2 Plan Aからの変更点

| 項目 | Plan A（合成データ） | Plan B（実データ） |
|------|---------------------|-------------------|
| データ生成方式 | 合成画像自動生成 | 実キャプチャ手動アノテーション |
| クラス数 | 8クラス | 4クラス |
| データ量 | 1000枚以上 | 100枚以下（PoC） |
| アノテーション | 自動（配置座標=ラベル） | 半自動（Roboflow） |
| Domain Gap | あり（合成vs実データ） | なし（実データ使用） |
| 作業時間 | スクリプト実行のみ | 半日程度の手作業 |

### 1.3 Plan Bが必要な理由

Plan Aで発生した問題:
1. **Domain Gap**: 合成データで学習したモデルが実際のミニマップで検出できない
2. **誤検出**: チャンピオンアイコンなどをwardとして検出

これらは**実データでのアノテーション**によって根本的に解決できます。

---

## 2. 検出クラス定義（4クラス）

| class_id | クラス名 | 説明 | ミニマップでの見た目 |
|----------|---------|------|-------------------|
| 0 | stealth_ward | 味方ステルスワード | 緑色の目アイコン |
| 1 | stealth_ward_enemy | 敵ステルスワード | 赤色の目アイコン |
| 2 | control_ward | 味方コントロールワード | ピンク/紫色のアイコン |
| 3 | control_ward_enemy | 敵コントロールワード | 赤ピンクのアイコン |

**注記**:
- 消滅間近（old）wardは通常wardと統合（視覚的差異が小さいため）
- ファーサイトワードはステルスワードと同色のため統合（赤チームでは区別不可）
- 敵味方の区別は色で判断（緑/青=味方、赤=敵）

---

## 3. 使用ツール

### 3.1 X-AnyLabeling（推奨）

**選定理由**:
- 完全ローカル動作（通信不要、オフラインで作業可能）
- YOLOフォーマット直接エクスポート
- SAM（Segment Anything Model）対応で半自動アノテーション可能
- pip一発でインストール可能
- 無料・オープンソース

**GitHub**: https://github.com/CVHub520/X-AnyLabeling

**インストール方法**:
```bash
# 仮想環境有効化
cd "c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL"
.venv\Scripts\activate

# CPU版（推奨、軽量）
pip install x-anylabeling-cvhub[cpu]

# GPU版（CUDA 12.x、SAM高速化）
pip install x-anylabeling-cvhub[cuda12]
```

**起動方法**:
```bash
xanylabeling
```

**主要機能**:
| 機能 | 説明 |
|------|------|
| バウンディングボックス | 矩形でオブジェクトを囲む |
| SAM支援 | クリックで自動セグメンテーション |
| YOLOエクスポート | labels/*.txt形式で直接保存 |
| ショートカット | W:矩形作成、A/D:画像移動 |

### 3.2 代替ツール: Roboflow

通信が安定している場合はRoboflowも選択肢:
- Web UIで直感的にアノテーション可能
- Auto Label機能あり
- 無料枠: Public project 10,000枚まで
- URL: https://roboflow.com/

### 3.3 学習環境

| 項目 | 推奨 |
|------|------|
| フレームワーク | YOLOv8（ultralytics） |
| Python | 3.10+ |
| GPU | NVIDIA RTX（CUDA対応） |
| 実行環境 | Jupyter Notebook |

---

## 4. データセット作成手順

### 4.1 画像収集

**ソース**: C:\dataset（既存のミニマップキャプチャ）

**選択基準**:
- wardが映っているフレームを優先
- 試合中盤以降（フレーム500-2500）を重点的に選択
- 複数の試合から満遍なく収集

**推奨画像数**:
| フェーズ | 画像数 | 目的 |
|---------|--------|------|
| PoC初期 | 30-50枚 | 最小限のテスト |
| PoC本番 | 80-100枚 | 精度評価 |
| 本番 | 300-500枚 | 高精度モデル（将来） |

**画像選択スクリプト**:
```python
import os
import random
import shutil
from pathlib import Path

# ソースディレクトリ
source_dir = Path(r"C:\dataset")
# 出力ディレクトリ
output_dir = Path(r"C:\dataset_annotation")
output_dir.mkdir(exist_ok=True)

# 全試合フォルダを取得
match_folders = list(source_dir.glob("JP1-*"))

# 各試合から5-10枚ずつランダム選択
selected_images = []
for match_folder in match_folders[:20]:  # 最大20試合
    frame_dir = match_folder / "0"
    if frame_dir.exists():
        frames = list(frame_dir.glob("*.png"))
        # フレーム500-2500の範囲でフィルタリング
        valid_frames = [f for f in frames if 500 <= int(f.stem) <= 2500]
        if valid_frames:
            selected = random.sample(valid_frames, min(5, len(valid_frames)))
            selected_images.extend(selected)

# 合計100枚まで制限
selected_images = selected_images[:100]

# コピー
for i, img_path in enumerate(selected_images):
    dest = output_dir / f"frame_{i:04d}.png"
    shutil.copy(img_path, dest)

print(f"選択完了: {len(selected_images)}枚")
```

### 4.2 X-AnyLabelingセットアップ

**Step 1: インストール**
```bash
cd "c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL"
.venv\Scripts\activate
pip install x-anylabeling-cvhub[cpu]
```

**Step 2: 起動**
```bash
xanylabeling
```

**Step 3: 画像フォルダを開く**
1. メニュー「File」→「Open Dir」
2. `C:\dataset_annotation` を選択
3. 画像一覧が左パネルに表示される

**Step 4: クラスを読み込む**
1. メニュー「File」→「Load Label File」
2. `C:\dataset_annotation\classes.txt` を選択
3. 4クラスが登録される

### 4.3 アノテーション作業

**クラス定義ファイル**（作成済み）:
`C:\dataset_annotation\classes.txt`
```
stealth_ward
stealth_ward_enemy
control_ward
control_ward_enemy
```

**Step 3: アノテーション実行**
1. `W`キーを押して矩形描画モードに入る
2. wardを矩形（Bounding Box）で囲む
3. クラス名を入力または選択
4. `Ctrl+S`で保存
5. `D`キーで次の画像へ

**ショートカットキー**:
| キー | 機能 |
|------|------|
| `W` | 矩形描画モード |
| `D` | 次の画像 |
| `A` | 前の画像 |
| `Ctrl+S` | 保存 |
| `Del` | 選択したラベル削除 |
| `Ctrl+Z` | 元に戻す |

**作業時間の目安**:
- 1枚あたり: 30秒-2分（ward数による）
- 100枚: 約2-3時間

---

## 5. アノテーションルール

### 5.1 バウンディングボックスの描き方

**正しい例**:
```
+----------+
|  [ward]  |  ← wardアイコン全体を囲む
+----------+
```

**間違った例**:
```
+----+
|ward|  ← 小さすぎる（アイコンの一部のみ）
+----+

+----------------+
|                |
|    [ward]      |  ← 大きすぎる（余白が多い）
|                |
+----------------+
```

### 5.2 判断基準

| 状況 | 対応 |
|------|------|
| wardが見える | アノテーション対象 |
| wardが半分以上隠れている | アノテーション対象外 |
| wardが30%程度隠れている | アノテーション対象（見えている部分で囲む） |
| 不明なアイコン | スキップ（アノテーションしない） |

### 5.3 クラス判別のポイント

**stealth_ward vs control_ward**:
- stealth_ward: 目のような形状
- control_ward: 花びらのような形状

**味方 vs 敵**:
- 味方: 緑/青/ピンク系の色
- 敵: 赤系の色

**farsight_wardについて**:
- ステルスワードと同色のため、stealth_wardとして扱う
- 赤チームでは区別不可能なため統合

### 5.4 注意事項

1. **チャンピオンアイコンをwardとしてラベル付けしない**
   - チャンピオンは円形、wardは独特の形状

2. **ピン（警戒ピン、危険ピンなど）をwardとしてラベル付けしない**
   - ピンは一時的に表示される
   - wardは固定位置に表示される

3. **タレットやインヒビターをwardとしてラベル付けしない**
   - これらは固定オブジェクトで形状が異なる

4. **wardが存在しないフレームも含める**
   - 負例（wardなし画像）は誤検出を減らすために重要
   - アノテーションなしで保存するだけでOK

---

## 6. データセット分割・エクスポート

### 6.1 X-AnyLabelingでのエクスポート

X-AnyLabelingはアノテーション時に自動でYOLO形式で保存されます。

**保存形式**:
- 画像: `C:\dataset_annotation\*.png`
- ラベル: `C:\dataset_annotation\labels\*.txt`

**YOLOラベル形式**:
```
<class_id> <x_center> <y_center> <width> <height>
```
- 座標は0-1に正規化された値

### 6.2 Train/Val/Test分割スクリプト

X-AnyLabelingはデータ分割機能がないため、Pythonスクリプトで分割します。

```python
import os
import random
import shutil
from pathlib import Path

# パス設定
SOURCE_DIR = Path(r"C:\dataset_annotation")
OUTPUT_DIR = Path(r"C:\dataset_annotation\split")

# 分割比率
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 出力ディレクトリ作成
for split in ['train', 'val', 'test']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

# 画像一覧取得
images = list(SOURCE_DIR.glob("*.png"))
random.shuffle(images)

# 分割
n = len(images)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:]
}

# コピー
for split, img_list in splits.items():
    for img_path in img_list:
        # 画像コピー
        shutil.copy(img_path, OUTPUT_DIR / 'images' / split / img_path.name)
        # ラベルコピー
        label_path = SOURCE_DIR / 'labels' / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, OUTPUT_DIR / 'labels' / split / label_path.name)

print(f"分割完了: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
```

### 6.3 data.yaml作成

YOLOv8学習用の設定ファイルを作成:

```yaml
# C:\dataset_annotation\split\data.yaml
path: C:/dataset_annotation/split
train: images/train
val: images/val
test: images/test

names:
  0: stealth_ward
  1: stealth_ward_enemy
  2: control_ward
  3: control_ward_enemy
```

**最終的なディレクトリ構造**:
```
C:\dataset_annotation\split\
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

---

## 7. YOLOv8学習手順

### 7.1 環境準備

```bash
# 仮想環境有効化
cd "c:\Users\lapis\Desktop\LoL_WorkSp_win\pyLoL-_WorkSp\pyLoL"
.venv\Scripts\activate

# PyTorch CUDA版インストール（未インストールの場合）
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ultralyticsインストール
python -m pip install ultralytics
```

### 7.2 学習実行

**Notebook: [7] ward_detection_planB.ipynb**

```python
# Cell [1]: 環境確認
import torch
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# Cell [2]: データセットパス設定
from pathlib import Path

# X-AnyLabelingでアノテーション後、分割スクリプトで作成したデータセット
DATASET_DIR = Path(r"C:\dataset_annotation\split")
DATA_YAML = DATASET_DIR / "data.yaml"

print(f"Dataset: {DATASET_DIR}")
print(f"data.yaml exists: {DATA_YAML.exists()}")
```

```python
# Cell [3]: YOLOv8学習
model = YOLO('yolov8n.pt')  # nano版（高速）

results = model.train(
    data=str(DATA_YAML),
    device='0',
    epochs=100,
    imgsz=512,
    batch=16,
    patience=20,

    # 小物体検出向け設定
    mosaic=1.0,
    copy_paste=0.3,

    # 反転無効化（ミニマップ方向性維持）
    flipud=0.0,
    fliplr=0.0,

    # プロジェクト設定
    project='models/ward_detector_planB',
    name='v1'
)
```

```python
# Cell [4]: 評価
metrics = model.val(data=str(DATA_YAML), split='test')

print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
```

```python
# Cell [5]: 推論テスト
test_image = r"C:\dataset\JP1-551272474\0\1500.png"

results = model.predict(
    source=test_image,
    conf=0.25,
    save=True,
    project='models/ward_detector_planB',
    name='predictions'
)

print(f"検出数: {len(results[0].boxes)}")
```

### 7.3 学習パラメータ解説

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| model | yolov8n.pt | 少量データではnano版が適切 |
| epochs | 100 | 十分な学習回数 |
| imgsz | 512 | ミニマップサイズに合わせる |
| batch | 16 | GPU VRAMに応じて調整 |
| patience | 20 | Early Stopping |
| mosaic | 1.0 | 小物体検出に効果的 |
| flipud/fliplr | 0.0 | ミニマップの方向性を保持 |

---

## 8. 評価基準

### 8.1 成功基準（PoC段階）

| 指標 | 目標値 | 備考 |
|------|--------|------|
| mAP@50 | ≥ 0.70 | 最低限の検出能力 |
| Precision | ≥ 0.70 | 誤検出が少ない |
| Recall | ≥ 0.60 | 検出漏れが許容範囲 |

### 8.2 評価方法

1. **定量評価**: mAP、Precision、Recallの数値確認
2. **定性評価**: 実際の推論結果を目視確認
   - wardが正しく検出されているか
   - チャンピオンアイコンを誤検出していないか
   - 検出漏れがないか

### 8.3 改善アクション

| 状況 | 対応 |
|------|------|
| mAP@50 < 0.50 | データ追加（50枚以上） |
| Precisionが低い | 誤検出パターンを分析し、負例を追加 |
| Recallが低い | 検出漏れパターンを分析し、該当wardを追加 |
| 特定クラスの精度が低い | 該当クラスのデータを重点的に追加 |

---

## 9. 実装スケジュール

### Phase 1: 環境準備（30分）
- [ ] Roboflowアカウント作成
- [ ] プロジェクト作成
- [ ] クラス定義

### Phase 2: データ収集（30分）
- [ ] C:\datasetから画像選択スクリプト実行
- [ ] 100枚の画像をC:\dataset_annotationに収集
- [ ] Roboflowにアップロード

### Phase 3: アノテーション（2-3時間）
- [ ] 100枚のアノテーション実行
- [ ] 品質チェック（ランダムサンプリング）

### Phase 4: 学習・評価（1時間）
- [ ] データセットエクスポート（YOLOv8形式）
- [ ] YOLOv8学習実行
- [ ] 評価・推論テスト

### Phase 5: 反復改善（必要に応じて）
- [ ] 精度不足の場合はデータ追加
- [ ] 誤検出パターンの分析と対策

---

## 10. トラブルシューティング

### 10.1 Roboflow関連

**問題**: 画像アップロードが遅い
- **対策**: 画像を事前にリサイズ（512x512）してからアップロード

**問題**: アノテーションが保存されない
- **対策**: ブラウザのキャッシュをクリア、別ブラウザで試す

### 10.2 学習関連

**問題**: CUDA out of memory
- **対策**: batch=8に減らす

**問題**: mAPが0.3以下
- **対策**:
  1. アノテーションの品質確認（バウンディングボックスが正確か）
  2. データ量の追加（50枚以上追加）
  3. epochs=200に増やす

**問題**: 特定クラスのAPが極端に低い
- **対策**: 該当クラスのデータを重点的に追加（クラス不均衡の解消）

### 10.3 推論関連

**問題**: 検出結果が表示されない
- **対策**: conf閾値を下げる（conf=0.1で試す）

**問題**: 誤検出が多い
- **対策**:
  1. conf閾値を上げる（conf=0.5で試す）
  2. 負例（wardなし画像）を追加して再学習

---

## 11. 次のステップ

### 11.1 PoC完了後

1. **データ拡張**
   - 100枚 → 300-500枚に増加
   - 多様な試合からデータ収集

2. **モデル改善**
   - yolov8n → yolov8s/m へアップグレード
   - ハイパーパラメータチューニング

3. **Auto Label活用**
   - PoC段階のモデルでAuto Label
   - 手動修正でデータ効率化

### 11.2 本番統合

1. **scraper.pyへの統合**
   - リアルタイムward検出機能
   - ward座標のCSV/JSON出力

2. **バッチ推論パイプライン**
   - C:\dataset全体（219,452フレーム）への適用
   - ward配置ヒートマップ生成

---

## 12. 参考リンク

### Roboflow
- 公式サイト: https://roboflow.com/
- ドキュメント: https://docs.roboflow.com/
- YOLOv8エクスポート: https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/

### YOLOv8
- 公式ドキュメント: https://docs.ultralytics.com/
- 学習ガイド: https://docs.ultralytics.com/modes/train/

### 関連ドキュメント（pyLoL内）
- Plan A仕様書: [ward_detection_implementation.md](ward_detection_implementation.md)
- 調査結果: [ward_detection_plan.md](ward_detection_plan.md)

---

## 13. 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2026-01-02 | 1.0 | 初版作成 |
| 2026-01-03 | 2.0 | PoC実装完了、進捗サマリー追加、v2モデル学習結果記載 |
| 2026-01-03 | 2.1 | v3モデル学習完了（150枚、mAP50=0.983）、control_ward_enemy精度大幅改善 |

---

**仕様書終わり**
