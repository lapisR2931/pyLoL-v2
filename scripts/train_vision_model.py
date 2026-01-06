"""
視界スコア予測モデルの学習スクリプト - Phase 5 Task C

使用方法:
    # ロジスティック回帰
    python scripts/train_vision_model.py --model logistic

    # CNN
    python scripts/train_vision_model.py --model cnn --epochs 50

    # 全データで学習（テスト分割なし）
    python scripts/train_vision_model.py --model logistic --no-split

    # データセットパス指定
    python scripts/train_vision_model.py --dataset data/vision_dataset.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoLeague.scoring.predictor import VisionPredictor, load_dataset


# =============================================================================
# 設定
# =============================================================================

DEFAULT_DATASET = PROJECT_ROOT / "data" / "vision_dataset.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# メイン処理
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="視界スコア予測モデル学習",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "cnn"],
        help="モデルタイプ (default: logistic)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="データセットパス (default: data/vision_dataset.npz)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="出力ディレクトリ (default: models/)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="CNNのエポック数 (default: 50)"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="データ分割せず全データで学習"
    )
    args = parser.parse_args()

    # パス設定
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("視界スコア予測モデル学習")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. データ読み込み
    # -------------------------------------------------------------------------
    print(f"\n[1] データ読み込み: {dataset_path}")

    if not dataset_path.exists():
        print(f"エラー: データセットが見つかりません: {dataset_path}")
        print("Task Bを先に実行してvision_dataset.npzを生成してください。")
        sys.exit(1)

    X, y, match_ids = load_dataset(dataset_path)

    print(f"    サンプル数: {len(X)}")
    print(f"    入力形状: {X.shape}")
    print(f"    ラベル分布: Blue勝利={y.sum()}, Red勝利={len(y)-y.sum()}")

    # -------------------------------------------------------------------------
    # 2. データ分割
    # -------------------------------------------------------------------------
    print("\n[2] データ分割")

    if args.no_split:
        X_train, X_test = X, X
        y_train, y_test = y, y
        print("    全データで学習（テスト分割なし）")
    else:
        # 層化抽出でtrain/test分割
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
        except ValueError:
            # サンプル数が少なすぎる場合は層化なしで分割
            print("    警告: サンプル数が少ないため層化抽出を無効化")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

        print(f"    学習データ: {len(X_train)}件")
        print(f"    テストデータ: {len(X_test)}件")

    # -------------------------------------------------------------------------
    # 3. モデル学習
    # -------------------------------------------------------------------------
    print(f"\n[3] モデル学習: {args.model}")

    predictor = VisionPredictor(model_type=args.model)

    if args.model == "cnn":
        metrics = predictor.fit(X_train, y_train, epochs=args.epochs, verbose=True)
    else:
        metrics = predictor.fit(X_train, y_train, verbose=True)

    # -------------------------------------------------------------------------
    # 4. 評価
    # -------------------------------------------------------------------------
    print("\n[4] 評価")

    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)

    # メトリクス計算
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics["test_accuracy"] = float(test_acc)
    metrics["test_precision"] = float(test_precision)
    metrics["test_recall"] = float(test_recall)
    metrics["test_f1"] = float(test_f1)
    metrics["n_samples_train"] = len(X_train)
    metrics["n_samples_test"] = len(X_test)

    print(f"    Accuracy:  {test_acc:.3f}")
    print(f"    Precision: {test_precision:.3f}")
    print(f"    Recall:    {test_recall:.3f}")
    print(f"    F1 Score:  {test_f1:.3f}")
    print(f"\n    混同行列:")
    print(f"    (予測→)   Red  Blue")
    print(f"    実際Red   {conf_matrix[0][0]:4d}  {conf_matrix[0][1]:4d}")
    print(f"    実際Blue  {conf_matrix[1][0]:4d}  {conf_matrix[1][1]:4d}")

    # -------------------------------------------------------------------------
    # 5. 保存
    # -------------------------------------------------------------------------
    print("\n[5] 保存")

    # モデル保存
    if args.model == "logistic":
        model_path = output_dir / "vision_predictor.joblib"
    else:
        model_path = output_dir / "vision_predictor.pt"

    predictor.save(model_path)
    print(f"    モデル: {model_path}")

    # メトリクス保存
    metrics_path = output_dir / "vision_predictor_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"    メトリクス: {metrics_path}")

    # 特徴量重要度保存
    importance = predictor.get_feature_importance()
    importance_path = output_dir / "vision_importance.npy"
    np.save(importance_path, importance)
    print(f"    特徴量重要度: {importance_path}")
    print(f"        形状: {importance.shape}")
    print(f"        値域: [{importance.min():.4f}, {importance.max():.4f}]")

    # -------------------------------------------------------------------------
    # 完了
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("学習完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
