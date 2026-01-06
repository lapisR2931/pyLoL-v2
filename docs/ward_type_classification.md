# Ward Type Classification (Riot Timeline API)

Riot APIのタイムラインデータにおけるwardType分類の調査結果。

## wardType一覧

| wardType | 説明 | 備考 |
|----------|------|------|
| `YELLOW_TRINKET` | 黄トリンケット（ステルスワード） | 無料、時間制限あり |
| `BLUE_TRINKET` | 青トリンケット（ファーサイトオルタレーション） | 遠距離設置可能、破壊されやすい |
| `CONTROL_WARD` | コントロールワード | 購入品、ステルス解除機能 |
| `SIGHT_WARD` | サポートアイテム由来のワード | サポートクエスト完了後 |
| `UNDEFINED` | **スキルによる視界確保** | チャンピオン固有スキル |

## UNDEFINEDについて

`UNDEFINED`はAPIのバグやデータ欠損ではなく、**チャンピオンスキルによる視界確保**を記録するためのwardTypeである。

### スキルで視界を提供するチャンピオン（確認済み）

| Champion | Role | スキル |
|----------|------|--------|
| Milio | Support | 視界提供スキル |
| Illaoi | Top | 視界提供スキル |
| Nidalee | Jungle | W（ブッシュワック） |
| Kayn | Jungle | 視界提供スキル |
| Maokai | Support/Top | E（サップリング・トス） |
| Teemo | Top | R（ネクソスキノコ） |
| Shaco | Jungle | W（ジャック・イン・ザ・ボックス） |
| Jhin | ADC | E（キャプティブ・オーディエンス） |
| Fiddlesticks | Jungle | パッシブ（エフィジー） |

※ 上記は一部。視界提供スキルを持つチャンピオンは他にも存在する。

## 調査データ

調査日: 2026-01-05
対象: JP1サーバー、10試合

| Match ID | Total Wards | UNDEFINED | 比率 |
|----------|-------------|-----------|------|
| JP1_555621265 | 192 | 43 | 22.4% |
| JP1_555639648 | 58 | 0 | 0.0% |
| JP1_555644427 | 98 | 0 | 0.0% |
| JP1_555650841 | 66 | 0 | 0.0% |
| JP1_555658734 | 72 | 12 | 16.7% |
| JP1_555675575 | 76 | 2 | 2.6% |
| JP1_555685439 | 145 | 80 | 55.2% |
| JP1_555689813 | 108 | 0 | 0.0% |
| JP1_555696773 | 87 | 0 | 0.0% |
| JP1_555711124 | 134 | 0 | 0.0% |

## 視界スコア指標への応用

Phase 5の視界スコア指標設計において、wardTypeを以下のように分類して扱うことを推奨：

1. **通常ward** - `YELLOW_TRINKET`, `BLUE_TRINKET`, `CONTROL_WARD`, `SIGHT_WARD`
2. **スキル視界** - `UNDEFINED`

スキル視界は通常wardとは異なる特性を持つため、別カテゴリとして評価する：
- 設置コストなし（スキルクールダウンのみ）
- 持続時間が異なる
- 破壊されにくい場合がある
- チャンピオン選択に依存

## 関連スクリプト

- `scripts/visualize_ward_timeline.py` - ward可視化
- `scripts/visualize_ward_by_player.py` - プレイヤー別ward分析
- `scripts/analyze_undefined_wards.py` - UNDEFINED分析
