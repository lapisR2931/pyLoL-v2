"""
5分刻み評価時点でのモデル学習・日本語グラフ生成スクリプト

使用方法:
    python scripts/run_5min_evaluation.py

出力:
    - data/prediction_dataset_5times.npz: データセット
    - results/accuracy_over_time_ja.png: 時間帯別精度推移グラフ（日本語）
    - results/contribution_over_time_ja.png: 貢献度推移グラフ（日本語）
    - results/model_comparison_5times.json: 結果JSON
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from autoLeague.prediction.feature_extractor import (
    build_prediction_dataset,
    load_prediction_dataset,
)
from autoLeague.prediction.baseline_predictor import train_all_models_multi_time
from autoLeague.prediction.evaluator import (
    visualize_accuracy_over_time_ja,
    visualize_contribution_over_time_ja,
    generate_report_multi_time,
)
from autoLeague.prediction.config import PREDICTION_TIMES_MS


def main():
    # パス設定
    DATA_DIR = PROJECT_ROOT / "data"
    TIMELINE_DIR = DATA_DIR / "timeline"
    MATCH_DIR = DATA_DIR / "match"
    DATASET_DIR = Path(r"C:\dataset_20260105")

    OUTPUT_DATASET = DATA_DIR / "prediction_dataset_5times.npz"
    MODEL_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("5分刻み評価時点でのモデル学習")
    print("=" * 60)
    print(f"評価時点: {[t // 60000 for t in PREDICTION_TIMES_MS]}分")

    # Step 1: データセット構築
    print("\n[Step 1] データセット構築")
    if not OUTPUT_DATASET.exists():
        print("データセットを構築中...")
        stats = build_prediction_dataset(
            timeline_dir=TIMELINE_DIR,
            match_dir=MATCH_DIR,
            dataset_dir=DATASET_DIR,
            output_path=OUTPUT_DATASET,
            times_ms=PREDICTION_TIMES_MS,
            verbose=True,
        )
        print(f"構築完了: {stats}")
    else:
        print(f"既存のデータセットを使用: {OUTPUT_DATASET}")

    # Step 2: データセット読み込み
    print("\n[Step 2] データセット読み込み")
    dataset = load_prediction_dataset(OUTPUT_DATASET)

    print(f"試合数: {len(dataset['match_ids'])}")
    print(f"評価時点: {[t // 60000 for t in dataset['times_ms']]}分")
    print(f"X_baseline shape: {dataset['X_baseline'].shape}")
    print(f"y shape: {dataset['y'].shape}")
    print(f"Blue勝利: {dataset['y'].sum()}, Red勝利: {len(dataset['y']) - dataset['y'].sum()}")

    # Step 3: 全時点・全モデル学習
    print("\n[Step 3] 全時点・全モデル学習")
    results_by_time = train_all_models_multi_time(
        dataset=dataset,
        output_dir=MODEL_DIR,
        vision_predictor=None,
        verbose=True,
        include_tactical=True,
    )

    # Step 4: レポート生成
    print("\n[Step 4] レポート生成")
    report = generate_report_multi_time(
        results_by_time=results_by_time,
        output_path=RESULTS_DIR / "model_comparison_5times.json",
    )
    print(report)

    # Step 5: 日本語グラフ生成
    print("\n[Step 5] 日本語グラフ生成")

    # 精度推移グラフ
    visualize_accuracy_over_time_ja(
        results_by_time=results_by_time,
        output_path=RESULTS_DIR / "accuracy_over_time_ja.png",
    )

    # 貢献度推移グラフ
    visualize_contribution_over_time_ja(
        results_by_time=results_by_time,
        output_path=RESULTS_DIR / "contribution_over_time_ja.png",
    )

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)
    print(f"\n出力ファイル:")
    print(f"  - データセット: {OUTPUT_DATASET}")
    print(f"  - 結果JSON: {RESULTS_DIR / 'model_comparison_5times.json'}")
    print(f"  - 精度推移グラフ: {RESULTS_DIR / 'accuracy_over_time_ja.png'}")
    print(f"  - 貢献度推移グラフ: {RESULTS_DIR / 'contribution_over_time_ja.png'}")


if __name__ == "__main__":
    main()
